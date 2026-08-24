# Bitcoin to Ethereum Cross-Chain Bridge

## Overview

A complete cross-chain bridge system that transfers Bitcoin to Ethereum as wrapped BTC (WBTC) with full validation and broadcasting on both networks.

## Features

✅ **Bitcoin to Ethereum Bridge**
- Lock Bitcoin on source chain
- Mint wrapped tokens on Ethereum
- Atomic swap guarantees
- Multi-signature security

✅ **Full Network Integration**
- Ethereum mainnet/testnet support
- Bitcoin mainnet/testnet support
- Multiple RPC providers for redundancy
- Automatic failover

✅ **Transaction Validation**
- Real-time Bitcoin blockchain validation
- Ethereum transaction confirmation tracking
- Configurable confirmation requirements
- Multi-API validation for reliability

✅ **SMTP Email Notifications**
- Transfer initiation alerts
- Bitcoin confirmation notifications
- Ethereum broadcast updates
- Completion confirmations
- Validation reports

## Components

### 1. Ethereum Network Connector (`ethereum_network_connector.py`)
- Connects to Ethereum network via multiple providers
- Handles transaction creation, signing, and broadcasting
- Real-time gas price estimation
- Transaction validation and confirmation tracking

### 2. Bitcoin-Ethereum Bridge (`bitcoin_ethereum_bridge.py`)
- Manages cross-chain token transfers
- Validates transactions on both chains
- Handles BTC to WBTC conversion
- Atomic swap logic
- Transaction logging and export

### 3. SMTP Notifier (`smtp_notifier.py`)
- Email notifications for all bridge events
- HTML-formatted emails
- Transaction status updates
- Validation reports

### 4. Bridge Orchestrator (`bridge_orchestrator.py`)
- Main entry point for bridge operations
- Coordinates entire transfer process
- CLI interface
- Batch operations support

## Installation

```bash
# Install dependencies
pip install -r bridge_requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Ethereum Configuration
ETH_PRIVATE_KEY=your_ethereum_private_key_here

# SMTP Configuration (optional)
SMTP_EMAIL=your_email@gmail.com
SMTP_PASSWORD=your_app_password_here
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

## Usage

### Basic Transfer

Transfer Bitcoin to Ethereum:

```bash
python bridge_orchestrator.py \
  --eth-address 0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d \
  --amount 0.1 \
  --network mainnet \
  --email your_email@example.com
```

### With Bitcoin Transaction Hash

If you already sent Bitcoin:

```bash
python bridge_orchestrator.py \
  --btc-address bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass \
  --eth-address 0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d \
  --amount 0.1 \
  --btc-tx-hash <your_bitcoin_tx_hash> \
  --network mainnet
```

### Check Balance

Check Ethereum address balance:

```bash
python bridge_orchestrator.py \
  --eth-address 0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d \
  --check-balance
```

### Validate Transactions

Validate all bridge transactions:

```bash
python bridge_orchestrator.py \
  --eth-address 0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d \
  --validate-only
```

## Python API Usage

```python
from decimal import Decimal
from bridge_orchestrator import BridgeOrchestrator

# Initialize orchestrator
orchestrator = BridgeOrchestrator(
    ethereum_network="mainnet",
    bitcoin_network="mainnet",
    notification_email="your_email@example.com"
)

# Execute transfer
success = orchestrator.execute_transfer(
    btc_source_address="bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass",
    eth_destination_address="0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d",
    amount_btc=Decimal("0.1"),
    eth_private_key="your_private_key_here",
    btc_tx_hash="optional_btc_tx_hash"
)

# Validate all transactions
summary = orchestrator.validate_all_and_notify()
print(summary)
```

## Transfer Process

1. **Initiation**
   - Create bridge transaction
   - Calculate WBTC amount (1:1 ratio minus 0.1% fee)
   - Generate unique bridge ID

2. **Bitcoin Validation** (if tx hash provided)
   - Validate transaction on Bitcoin blockchain
   - Check confirmations (minimum 6)
   - Verify amount and addresses

3. **Ethereum Transaction Creation**
   - Create transaction with current gas price
   - Estimate gas limit
   - Generate transaction object

4. **Broadcast to Ethereum**
   - Sign transaction with private key
   - Broadcast to Ethereum network
   - Monitor transaction hash

5. **Confirmation**
   - Wait for required confirmations (12 blocks)
   - Validate final transaction
   - Export transaction log

## Security Considerations

⚠️ **IMPORTANT SECURITY NOTES:**

1. **Private Keys**
   - Never commit private keys to version control
   - Use environment variables or secure key management
   - Consider hardware wallet integration for production

2. **Transaction Validation**
   - Always validate Bitcoin transactions before bridging
   - Wait for sufficient confirmations (6+ for BTC, 12+ for ETH)
   - Verify transaction details match expected values

3. **Network Selection**
   - Use testnet for testing
   - Verify network configuration before mainnet operations
   - Double-check all addresses

4. **Amount Verification**
   - Verify amounts before executing transfers
   - Understand bridge fees (0.1%)
   - Check gas prices and network congestion

## Network Configuration

### Mainnet
- **Ethereum**: Multiple RPC endpoints with automatic failover
- **Bitcoin**: Blockstream, BlockCypher, Blockchain.info APIs

### Testnet
- **Ethereum**: Sepolia, Goerli
- **Bitcoin**: Bitcoin Testnet

## Transaction Logs

All bridge transactions are logged to `bridge_transactions.json`:

```json
{
  "bridge_config": {
    "ethereum_network": "mainnet",
    "bitcoin_network": "mainnet",
    "min_btc_confirmations": 6,
    "min_eth_confirmations": 12
  },
  "transactions": [
    {
      "bridge_id": "abc123...",
      "source_chain": "Bitcoin",
      "destination_chain": "Ethereum",
      "btc_tx_hash": "...",
      "eth_tx_hash": "...",
      "status": "completed"
    }
  ]
}
```

## Monitoring

### Transaction Status

- `initiated`: Bridge transfer created
- `waiting_btc_confirmations`: Waiting for Bitcoin confirmations
- `processing_eth_transfer`: Creating Ethereum transaction
- `eth_broadcasted`: Broadcasted to Ethereum network
- `completed`: Successfully completed
- `failed`: Transfer failed

### Validation Status

- `pending`: Awaiting validation
- `btc_validation_failed`: Bitcoin validation failed
- `fully_validated`: Both chains validated
- `error`: Validation error

## Email Notifications

Email notifications are sent for:
- ✉️ Transfer initiation
- ✉️ Bitcoin confirmation
- ✉️ Ethereum broadcast
- ✉️ Transfer completion
- ✉️ Validation reports

Configure SMTP settings in `.env` file.

## Troubleshooting

### Connection Issues

If Ethereum connection fails:
- Check network connectivity
- Verify RPC endpoints are accessible
- Try alternative network (testnet)

### Transaction Not Confirming

If transaction stuck:
- Check gas price (may need to increase)
- Verify network is not congested
- Check block explorer for status

### Validation Fails

If validation fails:
- Verify transaction hash is correct
- Wait for more confirmations
- Check network selection (mainnet vs testnet)

## API Endpoints Used

### Bitcoin APIs
- Blockstream: `https://blockstream.info/api`
- BlockCypher: `https://api.blockcypher.com/v1/btc/main`
- Blockchain.info: `https://blockchain.info`

### Ethereum RPCs
- LlamaRPC: `https://eth.llamarpc.com`
- Ankr: `https://rpc.ankr.com/eth`
- PublicNode: `https://ethereum.publicnode.com`
- 1RPC: `https://1rpc.io/eth`

## Support

For issues or questions:
1. Check transaction logs in `bridge_transactions.json`
2. Review error messages in console output
3. Verify network and address configuration
4. Check blockchain explorers for transaction status

## License

Part of Nexus AGI project

---

**⚠️ DISCLAIMER**: This bridge system handles real cryptocurrency transactions. Always test on testnet first. Never share private keys. Understand the risks before using with real funds.
