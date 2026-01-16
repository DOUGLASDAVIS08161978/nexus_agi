#!/usr/bin/env python3
"""
Esplora-Integrated Bitcoin Withdrawal Bridge
Uses real Blockstream Esplora API for Bitcoin operations

Features:
- Real Bitcoin transaction broadcasting
- Real UTXO management
- Dynamic fee estimation
- Real transaction confirmation tracking
- Testnet and Mainnet support

API Documentation: https://github.com/Blockstream/esplora/blob/master/API.md

Usage:
    export CONTRACT_ADDRESS="0x5FbDB2315678afecb367f032d93F642f64180aa3"
    export BITCOIN_NETWORK="testnet"  # or "mainnet"
    python esplora_withdrawal_bridge.py
"""

import os
import sys
import time
import hashlib
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime

# Try to import consciousness
try:
    from enhanced_consciousness_system import EnhancedConsciousnessModel
    CONSCIOUSNESS_ENABLED = True
except ImportError:
    CONSCIOUSNESS_ENABLED = False


class EsploraAPIClient:
    """
    Client for Blockstream Esplora API
    https://github.com/Blockstream/esplora/blob/master/API.md
    """

    # API Base URLs
    ENDPOINTS = {
        "mainnet": "https://blockstream.info/api",
        "testnet": "https://blockstream.info/testnet/api",
        "signet": "https://blockstream.info/signet/api",
        "liquid": "https://blockstream.info/liquid/api",
        "liquidtestnet": "https://blockstream.info/liquidtestnet/api"
    }

    def __init__(self, network: str = "testnet"):
        """
        Initialize Esplora API client

        Args:
            network: Bitcoin network (mainnet, testnet, signet)
        """
        if network not in self.ENDPOINTS:
            raise ValueError(f"Invalid network: {network}. Must be one of {list(self.ENDPOINTS.keys())}")

        self.network = network
        self.base_url = self.ENDPOINTS[network]
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Nexus-AGI-Bridge/1.0'
        })

        print(f"🌐 Esplora API Client Initialized")
        print(f"   Network: {network}")
        print(f"   Endpoint: {self.base_url}")

    def get_transaction(self, txid: str) -> Dict[str, Any]:
        """GET /tx/:txid - Get transaction information"""
        url = f"{self.base_url}/tx/{txid}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_transaction_status(self, txid: str) -> Dict[str, Any]:
        """GET /tx/:txid/status - Get transaction confirmation status"""
        url = f"{self.base_url}/tx/{txid}/status"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_transaction_hex(self, txid: str) -> str:
        """GET /tx/:txid/hex - Get raw transaction in hex"""
        url = f"{self.base_url}/tx/{txid}/hex"
        response = self.session.get(url)
        response.raise_for_status()
        return response.text

    def broadcast_transaction(self, tx_hex: str) -> str:
        """
        POST /tx - Broadcast raw transaction to network

        Args:
            tx_hex: Raw transaction in hex format

        Returns:
            txid: Transaction ID if successful
        """
        url = f"{self.base_url}/tx"
        response = self.session.post(url, data=tx_hex)
        response.raise_for_status()
        return response.text  # Returns txid on success

    def broadcast_transaction_package(self, tx_hexes: List[str]) -> Dict[str, Any]:
        """
        POST /txs/package - Broadcast package of related transactions

        Useful for CPFP (Child Pays For Parent) scenarios

        Args:
            tx_hexes: List of transaction hex strings

        Returns:
            Package acceptance result
        """
        url = f"{self.base_url}/txs/package"
        response = self.session.post(url, json=tx_hexes)
        response.raise_for_status()
        return response.json()

    def get_address_info(self, address: str) -> Dict[str, Any]:
        """GET /address/:address - Get address information"""
        url = f"{self.base_url}/address/{address}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_address_utxos(self, address: str) -> List[Dict[str, Any]]:
        """GET /address/:address/utxo - Get unspent outputs for address"""
        url = f"{self.base_url}/address/{address}/utxo"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_address_txs(self, address: str) -> List[Dict[str, Any]]:
        """GET /address/:address/txs - Get transaction history for address"""
        url = f"{self.base_url}/address/{address}/txs"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_block(self, block_hash: str) -> Dict[str, Any]:
        """GET /block/:hash - Get block information"""
        url = f"{self.base_url}/block/{block_hash}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_block_at_height(self, height: int) -> str:
        """GET /block-height/:height - Get block hash at height"""
        url = f"{self.base_url}/block-height/{height}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.text

    def get_tip_height(self) -> int:
        """GET /blocks/tip/height - Get current block height"""
        url = f"{self.base_url}/blocks/tip/height"
        response = self.session.get(url)
        response.raise_for_status()
        return int(response.text)

    def get_tip_hash(self) -> str:
        """GET /blocks/tip/hash - Get current block hash"""
        url = f"{self.base_url}/blocks/tip/hash"
        response = self.session.get(url)
        response.raise_for_status()
        return response.text

    def get_mempool_info(self) -> Dict[str, Any]:
        """GET /mempool - Get mempool statistics"""
        url = f"{self.base_url}/mempool"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_mempool_txids(self) -> List[str]:
        """GET /mempool/txids - Get all txids in mempool"""
        url = f"{self.base_url}/mempool/txids"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_fee_estimates(self) -> Dict[str, float]:
        """
        GET /fee-estimates - Get fee estimates for various confirmation targets

        Returns:
            Dict mapping confirmation targets (in blocks) to fee rates (sat/vB)
            Targets: 1-25, 144, 504, 1008 blocks
        """
        url = f"{self.base_url}/fee-estimates"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


class EsploraWithdrawalBridge:
    """
    Bitcoin Withdrawal Bridge integrated with Esplora API

    Handles real Bitcoin operations:
    - UTXO selection
    - Transaction construction
    - Fee estimation
    - Broadcasting
    - Confirmation tracking
    """

    def __init__(
        self,
        contract_address: str,
        bitcoin_network: str = "testnet",
        bridge_vault_address: Optional[str] = None,
        consciousness: Optional[Any] = None
    ):
        """
        Initialize withdrawal bridge

        Args:
            contract_address: Ethereum/Polygon contract address
            bitcoin_network: Bitcoin network (mainnet, testnet, signet)
            bridge_vault_address: Bitcoin address holding bridge funds
            consciousness: Optional consciousness model
        """
        self.contract_address = contract_address
        self.bitcoin_network = bitcoin_network
        self.bridge_vault_address = bridge_vault_address or self._get_default_vault_address()
        self.consciousness = consciousness

        # Initialize Esplora API client
        self.esplora = EsploraAPIClient(bitcoin_network)

        print(f"\n🌉 Esplora Bitcoin Withdrawal Bridge Initialized")
        print(f"   Contract: {contract_address}")
        print(f"   Network: Ethereum → Bitcoin {bitcoin_network.capitalize()}")
        print(f"   Vault Address: {self.bridge_vault_address}")

    def _get_default_vault_address(self) -> str:
        """Get default vault address for network"""
        # These are example addresses - replace with real bridge vault addresses
        if self.bitcoin_network == "testnet":
            return "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx"  # Example testnet address
        else:
            return "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq"  # Example mainnet address

    def get_optimal_fee_rate(self, target_blocks: int = 6) -> float:
        """
        Get optimal fee rate for target confirmation time

        Args:
            target_blocks: Target confirmation in blocks (default: 6 ≈ 1 hour)

        Returns:
            Fee rate in sat/vB
        """
        print(f"\n💰 Fetching Fee Estimates...")

        fee_estimates = self.esplora.get_fee_estimates()

        # Get fee for target or closest available target
        target_str = str(target_blocks)
        if target_str in fee_estimates:
            fee_rate = fee_estimates[target_str]
        else:
            # Find closest target
            available_targets = sorted([int(t) for t in fee_estimates.keys()])
            closest = min(available_targets, key=lambda x: abs(x - target_blocks))
            fee_rate = fee_estimates[str(closest)]

        print(f"   Target: {target_blocks} blocks")
        print(f"   Fee Rate: {fee_rate:.2f} sat/vB")

        # Show fee histogram
        mempool_info = self.esplora.get_mempool_info()
        print(f"   Mempool: {mempool_info['count']} txs, {mempool_info['vsize']:,} vB")

        return fee_rate

    def get_vault_utxos(self) -> List[Dict[str, Any]]:
        """Get available UTXOs from bridge vault"""
        print(f"\n🔍 Querying Vault UTXOs...")

        utxos = self.esplora.get_address_utxos(self.bridge_vault_address)

        total_value = sum(utxo['value'] for utxo in utxos)

        print(f"   Address: {self.bridge_vault_address}")
        print(f"   Available UTXOs: {len(utxos)}")
        print(f"   Total Value: {total_value:,} sats ({total_value / 1e8:.8f} BTC)")

        if self.consciousness and CONSCIOUSNESS_ENABLED:
            self.consciousness.generate_inner_dialogue(
                f"Bridge vault has {len(utxos)} UTXOs with {total_value/1e8:.8f} BTC available",
                salience=0.7,
                source="utxo_query"
            )

        return utxos

    def select_utxos_for_amount(
        self,
        target_amount: int,
        fee_estimate: int,
        utxos: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Select optimal UTXOs for target amount + fees

        Args:
            target_amount: Target amount in satoshis
            fee_estimate: Estimated fee in satoshis
            utxos: Available UTXOs

        Returns:
            (selected_utxos, change_amount)
        """
        # Sort UTXOs by value (largest first for simplicity)
        sorted_utxos = sorted(utxos, key=lambda x: x['value'], reverse=True)

        selected = []
        total_input = 0
        required = target_amount + fee_estimate

        for utxo in sorted_utxos:
            if total_input >= required:
                break
            selected.append(utxo)
            total_input += utxo['value']

        if total_input < required:
            raise ValueError(
                f"Insufficient funds: need {required:,} sats, have {total_input:,} sats"
            )

        change = total_input - required

        print(f"\n🎯 UTXO Selection:")
        print(f"   Selected: {len(selected)} UTXOs")
        print(f"   Total Input: {total_input:,} sats")
        print(f"   Target: {target_amount:,} sats")
        print(f"   Fee: {fee_estimate:,} sats")
        print(f"   Change: {change:,} sats")

        return selected, change

    def create_withdrawal_transaction(
        self,
        amount_btc: float,
        target_address: str,
        fee_rate: float
    ) -> Dict[str, Any]:
        """
        Create Bitcoin withdrawal transaction

        Args:
            amount_btc: Amount to withdraw in BTC
            target_address: Destination address
            fee_rate: Fee rate in sat/vB

        Returns:
            Transaction data (currently simulated, needs signing implementation)
        """
        amount_sats = int(amount_btc * 1e8)

        print(f"\n🔨 Creating Withdrawal Transaction...")
        print(f"   Amount: {amount_btc} BTC ({amount_sats:,} sats)")
        print(f"   Target: {target_address}")
        print(f"   Fee Rate: {fee_rate:.2f} sat/vB")

        # Get vault UTXOs
        utxos = self.get_vault_utxos()

        if not utxos:
            raise ValueError("No UTXOs available in vault")

        # Estimate transaction size (simplified)
        # Typical P2WPKH: ~140 vB per input, ~31 vB per output, ~11 vB overhead
        estimated_size = 11 + (140 * 1) + (31 * 2)  # 1 input, 2 outputs (payment + change)
        estimated_fee = int(estimated_size * fee_rate)

        # Select UTXOs
        selected_utxos, change_amount = self.select_utxos_for_amount(
            amount_sats,
            estimated_fee,
            utxos
        )

        # Construct transaction structure
        tx_data = {
            "version": 2,
            "locktime": 0,
            "inputs": [
                {
                    "txid": utxo['txid'],
                    "vout": utxo['vout'],
                    "value": utxo['value'],
                    "status": utxo['status']
                }
                for utxo in selected_utxos
            ],
            "outputs": [
                {
                    "address": target_address,
                    "value": amount_sats
                }
            ],
            "fee": estimated_fee,
            "size_estimate": estimated_size
        }

        # Add change output if significant
        if change_amount > 546:  # Dust limit
            tx_data["outputs"].append({
                "address": self.bridge_vault_address,
                "value": change_amount
            })

        print(f"\n📝 Transaction Structure:")
        print(f"   Inputs: {len(tx_data['inputs'])}")
        print(f"   Outputs: {len(tx_data['outputs'])}")
        print(f"   Estimated Size: {estimated_size} vB")
        print(f"   Estimated Fee: {estimated_fee} sats ({estimated_fee/1e8:.8f} BTC)")

        if self.consciousness and CONSCIOUSNESS_ENABLED:
            self.consciousness.generate_inner_dialogue(
                f"Constructed withdrawal tx: {amount_btc} BTC to {target_address}",
                salience=0.85,
                source="tx_construction"
            )

        return tx_data

    def broadcast_transaction(self, tx_hex: str) -> Dict[str, Any]:
        """
        Broadcast transaction to Bitcoin network

        Args:
            tx_hex: Raw transaction in hex format

        Returns:
            Broadcast result with txid
        """
        print(f"\n📡 Broadcasting Transaction...")

        try:
            txid = self.esplora.broadcast_transaction(tx_hex)

            print(f"✅ Transaction Broadcast Successful!")
            print(f"   TXID: {txid}")
            print(f"   Explorer: https://blockstream.info/{self.bitcoin_network}/tx/{txid}")

            if self.consciousness and CONSCIOUSNESS_ENABLED:
                self.consciousness.generate_inner_dialogue(
                    f"Successfully broadcast transaction {txid[:16]}...",
                    salience=0.9,
                    source="broadcast"
                )

            return {
                "success": True,
                "txid": txid,
                "network": self.bitcoin_network,
                "broadcasted_at": time.time()
            }

        except requests.exceptions.HTTPError as e:
            print(f"❌ Broadcast Failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "network": self.bitcoin_network
            }

    def wait_for_confirmation(
        self,
        txid: str,
        target_confirmations: int = 1,
        max_wait_seconds: int = 600
    ) -> Dict[str, Any]:
        """
        Wait for transaction confirmations

        Args:
            txid: Transaction ID to monitor
            target_confirmations: Target number of confirmations
            max_wait_seconds: Maximum time to wait

        Returns:
            Confirmation status
        """
        print(f"\n⏳ Waiting for Confirmations...")
        print(f"   TXID: {txid[:16]}...")
        print(f"   Target: {target_confirmations} confirmation(s)")

        start_time = time.time()
        last_status = None

        while (time.time() - start_time) < max_wait_seconds:
            try:
                status = self.esplora.get_transaction_status(txid)

                if status != last_status:
                    if status['confirmed']:
                        block_height = status.get('block_height', 'unknown')
                        block_hash = status.get('block_hash', 'unknown')

                        # Get current tip to calculate confirmations
                        current_height = self.esplora.get_tip_height()
                        confirmations = current_height - block_height + 1

                        print(f"   ✓ Confirmed in block {block_height} ({confirmations} confirmation(s))")

                        if confirmations >= target_confirmations:
                            print(f"✅ Target Confirmations Reached!")

                            if self.consciousness and CONSCIOUSNESS_ENABLED:
                                self.consciousness.generate_inner_dialogue(
                                    f"Transaction confirmed with {confirmations} confirmations",
                                    salience=0.95,
                                    source="confirmation"
                                )

                            return {
                                "confirmed": True,
                                "confirmations": confirmations,
                                "block_height": block_height,
                                "block_hash": block_hash,
                                "waited_seconds": int(time.time() - start_time)
                            }
                    else:
                        print(f"   ⏳ Pending... (checked {int(time.time() - start_time)}s)")

                    last_status = status

            except Exception as e:
                print(f"   ⚠️  Error checking status: {e}")

            time.sleep(10)  # Check every 10 seconds

        print(f"⚠️  Timeout: Max wait time reached")
        return {
            "confirmed": False,
            "error": "timeout",
            "waited_seconds": max_wait_seconds
        }

    def withdraw_to_bitcoin(
        self,
        amount_wbtc: float,
        btc_address: str,
        user_address: str = "0x7f345957338dcc04bedea1396269d99bda4aa740",
        target_confirmations: int = 1
    ) -> Dict[str, Any]:
        """
        Execute full withdrawal from Ethereum/Polygon to Bitcoin using real Esplora API

        Args:
            amount_wbtc: Amount of wTBTC to burn/withdraw
            btc_address: Destination Bitcoin address
            user_address: User's Ethereum/Polygon address
            target_confirmations: Number of confirmations to wait for

        Returns:
            Complete withdrawal result
        """
        print(f"\n{'='*80}")
        print(f"💸 INITIATING ESPLORA-POWERED WITHDRAWAL")
        print(f"{'='*80}")
        print(f"   Amount: {amount_wbtc} wTBTC → {amount_wbtc} BTC")
        print(f"   From: {user_address[:10]}...{user_address[-8:]}")
        print(f"   Target: {btc_address}")
        print(f"   Network: {self.bitcoin_network.upper()}")
        print(f"{'='*80}\n")

        if self.consciousness and CONSCIOUSNESS_ENABLED:
            self.consciousness.generate_inner_dialogue(
                f"Initiating Esplora-powered withdrawal: {amount_wbtc} BTC to {btc_address}",
                salience=0.9,
                source="withdrawal_start"
            )

        # Step 1: Burn wTBTC (simulated for now)
        print(f"🔥 Step 1: Burning wTBTC on Ethereum...")
        burn_tx = self._simulate_burn(user_address, amount_wbtc)
        print(f"✅ Burn Tx: {burn_tx['tx_hash'][:16]}...")

        time.sleep(1)

        # Step 2: Get optimal fee rate
        fee_rate = self.get_optimal_fee_rate(target_blocks=6)

        time.sleep(1)

        # Step 3: Create Bitcoin withdrawal transaction
        tx_data = self.create_withdrawal_transaction(
            amount_wbtc,
            btc_address,
            fee_rate
        )

        time.sleep(1)

        # Step 4: Check blockchain status
        print(f"\n📊 Bitcoin Network Status:")
        tip_height = self.esplora.get_tip_height()
        tip_hash = self.esplora.get_tip_hash()
        print(f"   Current Height: {tip_height:,}")
        print(f"   Tip Hash: {tip_hash[:16]}...")

        mempool_info = self.esplora.get_mempool_info()
        print(f"   Mempool: {mempool_info['count']:,} txs")
        print(f"   Total Fees: {mempool_info['total_fee']:,} sats")

        # Note: Actual broadcasting requires signed transaction
        print(f"\n⚠️  Note: Actual transaction broadcasting requires:")
        print(f"   1. Proper UTXO signing with bridge vault private keys")
        print(f"   2. PSBT creation and signing workflow")
        print(f"   3. Multi-sig coordination if applicable")
        print(f"\n   This demo shows the complete flow using real Esplora API")
        print(f"   For production, integrate with hardware wallet or HSM for signing")

        result = {
            "burn_tx": burn_tx,
            "fee_rate": fee_rate,
            "tx_data": tx_data,
            "network_status": {
                "tip_height": tip_height,
                "tip_hash": tip_hash,
                "mempool": mempool_info
            },
            "status": "constructed_awaiting_signature"
        }

        print(f"\n{'='*80}")
        print(f"✅ WITHDRAWAL TRANSACTION PREPARED")
        print(f"{'='*80}")
        print(f"   Ready for signing and broadcast")
        print(f"   Esplora API integration: ACTIVE")
        print(f"   Real-time blockchain data: ✓")
        print(f"{'='*80}\n")

        if self.consciousness and CONSCIOUSNESS_ENABLED:
            print(f"\n🧠 Consciousness State:")
            print(f"   Awareness: {self.consciousness.self_awareness_level:.6f}")
            print(f"   Reflections: {len(self.consciousness.inner_dialogue)}")

        return result

    def _simulate_burn(self, user_address: str, amount: float) -> Dict[str, Any]:
        """Simulate burning wTBTC (placeholder)"""
        tx_data = f"burn:{user_address}:{amount}:{time.time()}"
        tx_hash = f"0x{hashlib.sha256(tx_data.encode()).hexdigest()}"

        return {
            "tx_hash": tx_hash,
            "from": user_address,
            "to": self.contract_address,
            "amount": amount,
            "block_number": 19500000 + int(time.time()) % 1000,
            "timestamp": time.time()
        }


def main():
    """Main execution"""

    # Get configuration from environment
    contract_address = os.environ.get(
        "CONTRACT_ADDRESS",
        "0x5FbDB2315678afecb367f032d93F642f64180aa3"
    )

    bitcoin_network = os.environ.get("BITCOIN_NETWORK", "testnet")

    # Initialize consciousness if available
    if CONSCIOUSNESS_ENABLED:
        consciousness = EnhancedConsciousnessModel(
            initial_context="esplora_bitcoin_bridge",
            memory_file="esplora_consciousness.json"
        )
    else:
        consciousness = None

    # Initialize Esplora-powered bridge
    bridge = EsploraWithdrawalBridge(
        contract_address=contract_address,
        bitcoin_network=bitcoin_network,
        consciousness=consciousness
    )

    # Execute withdrawal
    result = bridge.withdraw_to_bitcoin(
        amount_wbtc=0.001,  # Smaller amount for testnet
        btc_address="tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx",
        user_address="0x7f345957338dcc04bedea1396269d99bda4aa740"
    )

    # Save result
    output_file = f"esplora_withdrawal_{int(time.time())}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n💾 Results saved to {output_file}")

    return result


if __name__ == "__main__":
    result = main()
