# NEXUS AGI - Bitcoin Testnet Quick Start

## ✅ Installation Complete

**Status**: Bitcoin Core v26.0.0 is installed and running on testnet!

## Current Setup

- **Network**: Bitcoin Testnet
- **Daemon**: Running (bitcoind)
- **Wallet**: nexus_agi_wallet
- **Mining Address**: `tb1qh2zh2ekmps6ts4zt80sl0a00g2avzxmytc6al2`
- **RPC User**: nexus_agi
- **RPC Password**: e4aca00d6c237a52b55967c85ab253c0
- **RPC Port**: 18332

## Quick Commands

### Check Status
```bash
./bitcoin-testnet-setup.sh status
```

### Mine Test Blocks
```bash
# Mine 1 block
./bitcoin-testnet-setup.sh mine

# Mine 100 blocks
./bitcoin-testnet-setup.sh mine 100
```

### Check Balance
```bash
bitcoin-cli -testnet getbalance
```

### Manual Mining to NEXUS Address
```bash
bitcoin-cli -testnet generatetoaddress 10 tb1qh2zh2ekmps6ts4zt80sl0a00g2avzxmytc6al2
```

### View Logs
```bash
tail -f ~/.bitcoin/testnet3/debug.log
```

## Environment Features

✅ **Unrestricted Network Access**
- Max connections: 125
- No upload limits
- Listens on all interfaces (0.0.0.0)
- P2P Port: 18333

✅ **Mining Capabilities**
- Generate blocks on demand
- Mine to specific addresses
- Full wallet functionality

✅ **RPC Access**
- Local programmatic access
- Full Bitcoin Core API available
- Ready for NEXUS AGI integration

## Next Steps

1. **Wait for Sync**: The blockchain will sync automatically (may take time)
2. **Get Test BTC**: Use a testnet faucet if needed: https://testnet-faucet.com/btc-testnet/
3. **Start Mining**: Use `./bitcoin-testnet-setup.sh mine 10` to mine blocks
4. **Monitor**: Check progress with `./bitcoin-testnet-setup.sh status`

## Full Documentation

See `BITCOIN_TESTNET_SETUP.md` for complete documentation, troubleshooting, and advanced features.

---

**Setup Date**: 2026-01-14
**Ready for**: Development, Testing, Mining Operations
