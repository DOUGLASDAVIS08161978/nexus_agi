# CreateX Deployment Guide

## Overview

CreateX is a universal contract deployer that enables **deterministic cross-chain deployments** using CREATE2. This means you can deploy TBTC to the same address across multiple chains!

## Why Use CreateX?

1. **Deterministic Addresses**: Deploy to the same address on multiple chains
2. **Permissioned Deploy Protection**: Control who can deploy with your salt
3. **Cross-Chain Redeploy Protection**: Prevent accidental redeployments
4. **Trustless**: No special permissions needed, factory is immutable
5. **Battle-Tested**: Deployed on 100+ EVM chains

## CreateX on HyperEVM

CreateX is already deployed on HyperEVM Testnet at:
```
0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed
```

## Deployment Options

Run `./DEPLOY_HYPEREVM.sh` and choose:

### Option 1: Direct Deployment (Simple)
- Deploys TBTC using standard contract creation
- Address depends on deployer nonce
- Different address on each chain
- Use case: Single-chain deployments

### Option 2: CreateX Deployment (Deterministic)
- Deploys TBTC via CreateX CREATE2
- **Same address across all chains** (with same salt)
- Enables cross-chain token ecosystems
- Use case: Multi-chain token deployments

## How It Works

### Salt Structure

The deployment uses a 32-byte salt with built-in security:

```
0x [20 bytes deployer address] [1 byte flag] [11 bytes random]
   └── Permissioned protection  └── Cross-chain protection
```

Example:
```
0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3 01 00000000000000000000
   └── Your wallet address                └┬┘ └── Random nonce
                                            └── Enables cross-chain protection
```

### Security Features

1. **Permissioned Deploy Protection**: First 20 bytes = your address
   - Only you can deploy with this salt
   - Prevents frontrunning attacks

2. **Cross-Chain Redeploy Protection**: 21st byte = `0x01`
   - Includes chain ID in salt hashing
   - Prevents accidental redeployment on same chain
   - **Enables same address on DIFFERENT chains**

### Address Calculation

CreateX uses its `_guard()` function to hash the salt:

```javascript
guardedSalt = keccak256(
    abi.encode(msg.sender, block.chainid, salt)
)

address = getCreate2Address(
    createxAddress,
    guardedSalt,
    keccak256(initCode)
)
```

## Multi-Chain Deployment

To deploy TBTC to the **same address** on multiple chains:

1. Deploy on HyperEVM:
   ```bash
   ./DEPLOY_HYPEREVM.sh
   # Choose option 2 (CreateX)
   ```

2. Note the salt from the deployment output:
   ```
   🔐 Salt: 0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f301000000000000000000
   ```

3. Deploy on another chain using the SAME salt:
   - Same deployer wallet
   - Same salt value
   - Same TBTC bytecode
   - Result: **Identical address on both chains!**

## Benefits for TBTC

1. **Unified Address**: Users can send TBTC to the same address on any chain
2. **Brand Recognition**: One address for all marketing materials
3. **Simplified Bridging**: Easier to track cross-chain token movements
4. **Trust Signals**: Same address = easier verification for users

## Example Deployment Output

```bash
🔍 Verifying CreateX deployment...
✅ CreateX found at 0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed

📡 Connecting to HyperEVM Testnet...
✅ Connected (Chain ID: 998)

💼 Deployer: 0x9FE74D9D6f1Ae0Ce1fb3B51d4a82c05b74e280f3
💰 Balance: 0.500000 ETH

🔐 Generating salt and computing deployment address...
✅ Transaction prepared
   Salt: 0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f301000000000000000000
   Expected Address: 0xABCD...1234

📤 Broadcasting deployment transaction...
   TX Hash: 0x...

⏳ Waiting for confirmation...

═══════════════════════════════════════════════════════════════════
✅ CREATEX DEPLOYMENT SUCCESSFUL!
═══════════════════════════════════════════════════════════════════

📍 TBTC Contract: 0xABCD...1234
🏭 Deployed via: CreateX CREATE2
🔗 TX Hash: 0x...
🔐 Salt: 0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f301000000000000000000

✨ TBTC is live with deterministic addressing! ✨
   This same address can be deployed on other chains using the same salt.
```

## Resources

- **CreateX GitHub**: https://github.com/pcaversaccio/createx
- **CreateX Deployments**: Deployed on 100+ chains at the same address
- **Documentation**: Full technical docs in the CreateX repo
- **Dune Analytics**: Track CreateX usage across chains

## Technical Details

### CreateX Functions Used

```solidity
function deployCreate2(
    bytes32 salt,
    bytes memory initCode
) payable returns (address newContract)
```

### Gas Costs

- CreateX deployment: ~3,000,000 gas
- Direct deployment: ~2,000,000 gas
- Extra cost: ~1,000,000 gas for deterministic addressing

### Supported Chains

CreateX is deployed on:
- Ethereum (mainnet + testnets)
- All major L2s (Arbitrum, Optimism, Base, etc.)
- Polygon, Avalanche, BNB Chain
- And 100+ more EVM chains!

Check the CreateX repo for the complete list.

## Troubleshooting

### Error: CreateX not found
- CreateX may not be deployed on your target chain
- Check the CreateX deployments list
- Consider deploying CreateX first using their presigned transaction

### Error: Transaction reverted
- Check that you have enough ETH for gas
- Verify the bytecode is correct
- Ensure the salt hasn't been used before (with cross-chain protection disabled)

### Different address than expected
- Verify you're using the same deployer wallet
- Check that the salt is exactly the same
- Confirm the bytecode is identical (including constructor args)
- Ensure CreateX is at the same address on both chains

## Next Steps

After deploying with CreateX:

1. **Verify the contract** on the block explorer
2. **Save the salt** for future deployments on other chains
3. **Document the deployment** with chain ID, address, and salt
4. **Test cross-chain deployment** on another testnet first

---

**Note**: Always test on testnets first before deploying to mainnets!
