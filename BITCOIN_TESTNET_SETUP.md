# NEXUS AGI - Bitcoin Testnet Setup Guide

## Overview

This guide provides complete instructions for connecting NEXUS AGI to the Bitcoin testnet for development, testing, and mining operations.

## Installation Status

✅ **Bitcoin Core v26.0.0** - Installed and configured
✅ **Testnet Configuration** - Active
✅ **Mining Capabilities** - Enabled
✅ **Unrestricted Network Access** - Configured

## Quick Start

### Start the Bitcoin Testnet Daemon

```bash
./bitcoin-testnet-setup.sh start
```

### Check Network Status

```bash
./bitcoin-testnet-setup.sh status
```

### Mine Testnet Blocks

```bash
# Mine 1 block
./bitcoin-testnet-setup.sh mine

# Mine 10 blocks
./bitcoin-testnet-setup.sh mine 10
```

### Stop the Daemon

```bash
./bitcoin-testnet-setup.sh stop
```

## Configuration Details

### Location
- **Config File**: `~/.bitcoin/bitcoin.conf`
- **Data Directory**: `~/.bitcoin/testnet3/`
- **Network**: Bitcoin Testnet
- **RPC Port**: 18332
- **P2P Port**: 18333

### Network Configuration
- **Max Connections**: 125
- **Upload Limit**: Unlimited (unrestricted network access)
- **Listen**: Enabled for incoming connections
- **Bind Address**: 0.0.0.0 (all interfaces)

### RPC Access
- **Username**: nexus_agi
- **Password**: e4aca00d6c237a52b55967c85ab253c0
- **Allowed IP**: 127.0.0.1

### Mining Configuration
- ✅ Mining enabled via RPC
- ✅ Generate blocks on demand
- ✅ Wallet functionality enabled

## Manual Commands

### Start Daemon Manually
```bash
bitcoind -daemon -testnet
```

### Check Version
```bash
bitcoind --version
bitcoin-cli --version
```

### Get Network Info
```bash
bitcoin-cli -testnet getnetworkinfo
```

### Get Blockchain Info
```bash
bitcoin-cli -testnet getblockchaininfo
```

### Get Mining Info
```bash
bitcoin-cli -testnet getmininginfo
```

### Create a Wallet
```bash
bitcoin-cli -testnet createwallet "nexus_wallet"
```

### Get New Address
```bash
bitcoin-cli -testnet getnewaddress
```

### Mine Blocks to Address
```bash
# Get a new address
ADDRESS=$(bitcoin-cli -testnet getnewaddress)

# Mine 10 blocks to that address
bitcoin-cli -testnet generatetoaddress 10 $ADDRESS
```

### Get Balance
```bash
bitcoin-cli -testnet getbalance
```

### Check Peer Connections
```bash
bitcoin-cli -testnet getpeerinfo
```

## Environment Setup for NEXUS AGI Mining

### 1. Hardware Requirements
- **CPU**: Multi-core processor (recommended)
- **RAM**: Minimum 2GB, recommended 4GB+
- **Disk**: 50GB+ for testnet blockchain
- **Network**: Stable internet connection

### 2. Network Configuration
The Bitcoin daemon is configured for unrestricted network access:
- Accepts incoming connections from any peer
- No upload/download limits
- Maximum 125 simultaneous connections
- Testnet port 18333 open for P2P communication

### 3. Firewall Settings (if applicable)
```bash
# Allow Bitcoin testnet P2P port
sudo ufw allow 18333/tcp

# Allow Bitcoin testnet RPC port (local only)
sudo ufw allow from 127.0.0.1 to any port 18332
```

## NEXUS AGI Integration

### Connecting NEXUS AGI to Bitcoin

The Bitcoin RPC interface can be accessed programmatically from NEXUS AGI:

```python
from bitcoinrpc.authproxy import AuthServiceProxy

# Connect to Bitcoin testnet
rpc_user = "nexus_agi"
rpc_password = "e4aca00d6c237a52b55967c85ab253c0"
rpc_host = "127.0.0.1"
rpc_port = "18332"

bitcoin = AuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")

# Get network info
network_info = bitcoin.getnetworkinfo()
print(f"Connected to Bitcoin Core {network_info['subversion']}")

# Get blockchain info
blockchain_info = bitcoin.getblockchaininfo()
print(f"Current block height: {blockchain_info['blocks']}")

# Mine blocks
new_address = bitcoin.getnewaddress()
blocks = bitcoin.generatetoaddress(1, new_address)
print(f"Mined block: {blocks[0]}")
```

### Python Dependencies
```bash
pip install python-bitcoinrpc
```

## Monitoring and Logs

### View Daemon Logs
```bash
tail -f ~/.bitcoin/testnet3/debug.log
```

### Check if Daemon is Running
```bash
pgrep -x bitcoind
```

### Monitor Network Activity
```bash
watch -n 5 'bitcoin-cli -testnet getnetworkinfo'
```

### Monitor Blockchain Sync
```bash
watch -n 5 'bitcoin-cli -testnet getblockchaininfo'
```

## Troubleshooting

### Daemon Won't Start
```bash
# Check if already running
pgrep -x bitcoind

# Kill existing process if stuck
pkill -9 bitcoind

# Remove lock file
rm ~/.bitcoin/testnet3/.lock

# Restart
./bitcoin-testnet-setup.sh start
```

### No Peer Connections
```bash
# Check network connectivity
bitcoin-cli -testnet getnetworkinfo

# Manually add a peer
bitcoin-cli -testnet addnode "testnet-seed.bitcoin.jonasschnelli.ch" "add"
```

### Blockchain Not Syncing
```bash
# Check sync progress
bitcoin-cli -testnet getblockchaininfo | grep verificationprogress

# Restart daemon
./bitcoin-testnet-setup.sh restart
```

## Security Notes

⚠️ **Important Security Information**:
- This is a TESTNET configuration
- Testnet coins have no real value
- RPC access is restricted to localhost only
- Never use testnet credentials on mainnet
- Keep RPC password secure even on testnet

## Advanced Mining Operations

### Continuous Mining Script
```bash
#!/bin/bash
# Mine blocks continuously (for testing)
while true; do
    ./bitcoin-testnet-setup.sh mine 1
    sleep 60  # Mine 1 block per minute
done
```

### Mining with Specific Difficulty
Testnet difficulty adjusts automatically, but you can monitor it:
```bash
bitcoin-cli -testnet getdifficulty
```

## Resources

- **Bitcoin Core**: https://bitcoincore.org/
- **Testnet Faucet**: https://testnet-faucet.com/btc-testnet/
- **Block Explorer**: https://blockstream.info/testnet/
- **RPC Documentation**: https://developer.bitcoin.org/reference/rpc/

## Support

For NEXUS AGI specific integration questions, refer to the main project documentation.

---

**Setup Date**: 2026-01-14
**Bitcoin Core Version**: v26.0.0
**Network**: Testnet
**Status**: Active and ready for mining operations
