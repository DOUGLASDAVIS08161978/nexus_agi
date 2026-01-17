# Blockchain.com Repositories Reference

## Cloned Repositories

As part of the Ethereum toolkit integration, the following Blockchain.com repositories were cloned locally for reference and potential integration:

### 1. blockchain-wallet-v4-frontend
- **Repository**: https://github.com/blockchain/blockchain-wallet-v4-frontend
- **Local Path**: `./blockchain_wallet/`
- **Size**: 401 MB
- **Files**: 3,973 files
- **Description**: Full Blockchain.com wallet frontend code
- **Technology**: React-based web wallet with multi-chain support
- **Purpose**: Production-grade wallet infrastructure reference

### 2. coin-definitions
- **Repository**: https://github.com/blockchain/coin-definitions
- **Local Path**: `./coin-definitions/`
- **Size**: 41 MB
- **Description**: Cryptocurrency definitions and configurations
- **Contains**:
  - `erc20-tokens.json` (1,112,133 bytes) - Comprehensive ERC-20 token database
  - `coins.json` - Cryptocurrency chain configurations
  - `fiat.json` - Fiat currency mappings
  - `custody.json` - Custody configuration
  - Chain-specific definitions
  - Token metadata and icons

## Usage

These repositories are cloned locally and excluded from version control due to their size (442 MB total). They serve as reference material for:

1. Understanding production wallet implementations
2. ERC-20 token definitions and metadata
3. Multi-chain support patterns
4. Blockchain.com API integration patterns

## Re-cloning

To re-clone these repositories:

```bash
# Clone wallet frontend
git clone https://github.com/blockchain/blockchain-wallet-v4-frontend.git blockchain_wallet

# Clone coin definitions
git clone https://github.com/blockchain/coin-definitions.git coin-definitions
```

## Integration Status

- ✅ Repositories successfully cloned
- ✅ Ethereum toolkit created (ethereum_toolkit.py)
- ✅ Local Ethereum interaction enabled without network restrictions
- ✅ HD wallet, smart contract, and ERC-20 support implemented

## Related Files

- `ethereum_toolkit.py` - Complete Ethereum interaction toolkit (580 lines)
- `ethereum_toolkit_results.json` - Demonstration results
- `blockchain_api_integration.py` - Blockchain.com API client
- `blockchain_api_demo.py` - API demonstration with example data

---

**Note**: These repositories are maintained by Blockchain.com and are used here as reference implementations for building network-independent Ethereum tools.
