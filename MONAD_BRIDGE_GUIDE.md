# Bitcoin Regtest to Monad Testnet Bridge - Complete Guide

**Author:** Douglas Shane Davis & Claude
**Date:** 2026-01-22
**Status:** ✅ Operational

---

## Overview

This system provides a complete integration for:
1. Mining Bitcoin blocks on regtest (local testing network)
2. Bridging mined BTC to Monad testnet as WBTC
3. Signing and broadcasting transactions with your private key
4. Depositing tokens to your receiving address

---

## Configuration

### Your Monad Account Details

```
Private Key: c411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9
Signing Address: 0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771
WBTC Receiving: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
```

### Monad Testnet Configuration

```
RPC Endpoint: https://testnet-rpc.monad.xyz
Chain ID: 10143
Network: Monad Testnet
```

---

## Quick Start

### 1. Run the Bridge

```bash
# Navigate to project directory
cd /home/user/nexus_agi

# Run the bridge
python3 monad_regtest_bridge.py
```

### 2. Using the Launcher Script

```bash
# Make executable
chmod +x launch_monad_bridge.sh

# Run
./launch_monad_bridge.sh
```

---

## What the System Does

### 🔷 Bitcoin Regtest Mining

The system simulates Bitcoin regtest mining with:
- **Block Reward:** 50 BTC per block
- **Difficulty:** `0000` prefix (fast mining)
- **Proof of Work:** SHA-256 hash mining
- **Chain Tracking:** Complete blockchain history

Example output:
```
⛏️  Mining regtest block #1...
✅ Block mined in 0.06s
   Hash: 0000ac754fa3480b33c7516f52664719...
   Nonce: 61,776
   Reward: 50.0 BTC
```

### 🌉 Monad Testnet Bridge

After every N blocks (default: 2), the system:
1. Calculates accumulated BTC
2. Creates bridge transaction (1:1 ratio BTC → WBTC)
3. Signs transaction with your private key
4. Broadcasts to Monad testnet
5. Waits for confirmation

Example output:
```
🌉 BRIDGING TO MONAD TESTNET
Bridge ID: BRIDGE-1769091920-270421f9
BTC Amount: 100.00000000 BTC
WBTC Amount: 100.00000000 WBTC
From: 0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771
To: 0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771

🔏 Signing transaction...
📤 Broadcasting transaction...
✅ Transaction confirmed!
   TX Hash: 0xabc123...
```

---

## Session Results

After each session, you get:

### 📊 Mining Statistics
- Blocks mined
- Total BTC earned
- Chain length

### 🌉 Bridge Statistics
- Number of bridge transactions
- Total WBTC bridged
- Receiving address
- Transaction hashes

### 💾 JSON Export
All session data is exported to:
```
monad_regtest_session_<timestamp>.json
```

Example structure:
```json
{
  "session_id": "REGTEST-SESSION-1769091922",
  "timestamp": "2026-01-22T14:25:22",
  "mining": {
    "blocks_mined": 10,
    "total_btc": 500.0,
    "blocks": [...]
  },
  "bridging": {
    "transactions": 5,
    "total_wbtc": 500.0,
    "receiving_address": "0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771",
    "bridges": [...]
  }
}
```

---

## Getting Testnet Funds

### Issue: Insufficient Balance

If you see this error:
```
❌ Error bridging to Monad: {'code': -32603, 'message': 'Signer had insufficient balance'}
```

This means your account needs testnet ETH for gas fees.

### Solutions:

#### Option 1: Monad Testnet Faucet
Visit the official Monad testnet faucet:
- https://faucet.monad.xyz (check if available)
- https://testnet.monad.xyz/faucet
- Request testnet ETH for address: `0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771`

#### Option 2: Bridge from Other Testnets
If Monad has a bridge from Sepolia or other testnets:
1. Get Sepolia ETH from https://sepoliafaucet.com/
2. Bridge to Monad testnet

#### Option 3: Community Resources
- Ask in Monad Discord/Telegram for testnet funds
- Check Monad documentation for faucet alternatives

---

## Customization

### Change Mining Parameters

Edit `monad_regtest_bridge.py`:

```python
# In main() function

# Mine more/fewer blocks
session_data = system.run_mining_and_bridge_session(
    blocks_to_mine=20,          # Change this
    bridge_every_n_blocks=5      # Bridge less frequently
)
```

### Change Block Reward

```python
# At top of file
REGTEST_BLOCK_REWARD = 100.0  # Increase block reward
```

### Change Difficulty

```python
# At top of file
REGTEST_DIFFICULTY = "00000"  # More zeros = harder (slower)
```

### Change Network

```python
# Use different network
MONAD_TESTNET_RPC = "https://your-custom-rpc.com"
MONAD_CHAIN_ID = 12345
```

---

## Advanced Usage

### Programmatic Access

```python
from monad_regtest_bridge import RegtestToMonadSystem

# Initialize system
system = RegtestToMonadSystem(
    monad_private_key="your_private_key",
    wbtc_receiving_address="0x..."
)

# Mine and bridge
session = system.run_mining_and_bridge_session(
    blocks_to_mine=5,
    bridge_every_n_blocks=1
)

# Check balance
balance = system.check_receiving_address_balance()

# Export session
filename = system.export_session(session)
```

### Accessing Individual Components

```python
# Just mining (no bridge)
from monad_regtest_bridge import RegtestMiner

miner = RegtestMiner()
blocks = miner.mine_blocks(10)
print(f"Mined {len(blocks)} blocks!")
```

```python
# Just bridging (no mining)
from monad_regtest_bridge import MonadTestnetBridge

bridge = MonadTestnetBridge(
    private_key="...",
    receiving_address="0x..."
)

tx = bridge.bridge_btc_to_wbtc(
    btc_amount=50.0,
    btc_txid="0000abc..."
)
```

---

## Troubleshooting

### Connection Issues

**Problem:** Cannot connect to Monad testnet

**Solutions:**
1. Check if RPC is correct: `https://testnet-rpc.monad.xyz`
2. Test connection:
   ```bash
   curl -X POST https://testnet-rpc.monad.xyz \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
   ```
3. Try alternative RPC endpoints
4. Check if testnet is operational

### Private Key Issues

**Problem:** Invalid private key or signing errors

**Solutions:**
1. Verify private key format (64 hex characters, no 0x prefix)
2. Check that private key matches expected address
3. Test with Web3:
   ```python
   from eth_account import Account
   account = Account.from_key("your_key")
   print(account.address)
   ```

### Gas Price Too High

**Problem:** Transaction gas price is too expensive

**Solution:** Modify gas settings in code:
```python
tx = {
    'gas': 21000,
    'gasPrice': self.w3.to_wei(50, 'gwei'),  # Lower gas price
    # ... rest of tx
}
```

---

## Security Notes

⚠️ **IMPORTANT SECURITY CONSIDERATIONS:**

1. **Private Key Storage**
   - Private key is hardcoded for testnet use only
   - NEVER use this private key on mainnet
   - NEVER commit private keys to git in production

2. **Testnet Only**
   - This system is designed for TESTNET ONLY
   - Do not send real funds
   - Testnet tokens have NO VALUE

3. **RPC Security**
   - Using public RPC endpoints
   - Consider running your own node for production
   - Rate limits may apply

---

## System Architecture

```
┌─────────────────────┐
│  Regtest Miner      │
│  - Mine blocks      │
│  - SHA-256 PoW      │
│  - Chain tracking   │
└──────────┬──────────┘
           │
           │ BTC rewards
           ▼
┌─────────────────────┐
│  Bridge System      │
│  - Convert BTC→WBTC │
│  - Sign with key    │
│  - Track history    │
└──────────┬──────────┘
           │
           │ Signed transactions
           ▼
┌─────────────────────┐
│  Monad Testnet      │
│  - Receive tx       │
│  - Execute          │
│  - Confirm          │
└──────────┬──────────┘
           │
           │ WBTC tokens
           ▼
┌─────────────────────┐
│  Your Address       │
│  0x24f6b1ce...      │
│  - Hold WBTC        │
│  - Use in DeFi      │
└─────────────────────┘
```

---

## Next Steps

Once you have testnet funds:

1. **Verify Transactions**
   - Check Monad testnet explorer
   - View transaction history
   - Confirm WBTC balance

2. **Use WBTC**
   - Interact with DeFi protocols on Monad
   - Swap, stake, or provide liquidity
   - Test protocol integrations

3. **Automate**
   - Set up cron job for regular mining
   - Create continuous bridge operation
   - Build monitoring dashboards

4. **Scale Up**
   - Increase block mining
   - Add multiple receiving addresses
   - Implement more sophisticated mining strategies

---

## File Structure

```
nexus_agi/
├── monad_regtest_bridge.py       # Main bridge system
├── launch_monad_bridge.sh        # Launcher script
├── MONAD_BRIDGE_GUIDE.md         # This file
└── monad_regtest_session_*.json  # Session exports
```

---

## Resources

- **Monad Documentation:** https://docs.monad.xyz
- **Monad Testnet:** https://testnet.monad.xyz
- **Web3.py Docs:** https://web3py.readthedocs.io
- **Bitcoin Regtest:** https://developer.bitcoin.org/examples/testing.html

---

## Support

For issues or questions:
1. Check this guide first
2. Review session JSON for transaction details
3. Check Monad testnet status
4. Review code comments in `monad_regtest_bridge.py`

---

## Changelog

### 2026-01-22 - v1.0
- ✅ Initial release
- ✅ Bitcoin regtest mining
- ✅ Monad testnet integration
- ✅ Transaction signing and broadcasting
- ✅ Session export and tracking
- ✅ Balance checking
- ✅ Comprehensive error handling

---

**Happy Mining! ⛏️🌉**
