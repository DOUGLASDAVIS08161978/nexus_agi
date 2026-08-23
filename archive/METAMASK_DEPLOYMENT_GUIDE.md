# MetaMask/ZetaLink Deployment Guide

## Overview

This guide explains how to deploy Nexus AGI Bridge contracts using MetaMask instead of providing a raw private key.

## Prerequisites

1. **MetaMask Browser Extension**
   - Install from: https://metamask.io
   - Create or import wallet
   - Address: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771

2. **ZetaLink Snap** (Optional - for Bitcoin features)
   - Installed via the deployment interface
   - Version: 0.2.1
   - Package: npm:zetalink

3. **Native Tokens for Gas**
   - **Sepolia:** ~0.1 ETH (get from faucets)
   - **Polygon:** ~0.5 MATIC (purchase or faucet)

## Deployment Methods

### Method 1: Browser-Based Deployment (Recommended)

1. Open `deploy_metamask.html` in your browser
2. Click "Connect MetaMask"
3. Approve the connection in MetaMask
4. Select your network (Sepolia or Polygon)
5. Click "Deploy All (Recommended)"
6. Approve each transaction in MetaMask

### Method 2: Hardhat with MetaMask RPC

```bash
# Set MetaMask RPC in .env
SEPOLIA_RPC_URL=http://localhost:8545

# Connect MetaMask to localhost:8545
# Then deploy
npx hardhat run scripts/deploy_testnet_bridge.js --network sepolia
```

### Method 3: Manual Contract Interaction

Use MetaMask's contract interaction feature:
1. Go to Remix IDE: https://remix.ethereum.org
2. Load contracts from `contracts/` directory
3. Compile contracts
4. Deploy using MetaMask

## ZetaLink Features

### Bitcoin Wallet Derivation

```javascript
const btcWallet = await window.ethereum.request({
  method: 'wallet_snap',
  params: {
    snapId: 'npm:zetalink',
    request: {
      method: 'derive-btc-wallet',
      params: [false] // false = testnet, true = mainnet
    }
  }
});
```

### Get Bitcoin UTXOs

```javascript
const utxos = await window.ethereum.request({
  method: 'wallet_snap',
  params: {
    snapId: 'npm:zetalink',
    request: {
      method: 'get-btc-utxo',
      params: []
    }
  }
});
```

### Cross-Chain Bitcoin Transaction

```javascript
const txHash = await window.ethereum.request({
  method: 'wallet_snap',
  params: {
    snapId: 'npm:zetalink',
    request: {
      method: 'transact-btc',
      params: [
        customMemo,
        depositFee,
        recipientAddress,
        ZRC20ContractAddress,
        amount
      ]
    }
  }
});
```

## Security Notes

✅ **Advantages of MetaMask Deployment:**
- No need to expose private key
- Transaction approval in secure MetaMask interface
- Hardware wallet support (Ledger, Trezor)
- Better security practices

⚠️ **Important:**
- Always verify contract addresses
- Check transaction details before approving
- Use testnet first
- Never share recovery phrase

## Troubleshooting

### MetaMask Not Detected
- Ensure MetaMask extension is installed
- Refresh the page
- Check browser console for errors

### Transaction Fails
- Check gas balance
- Verify network selection
- Increase gas limit if needed

### ZetaLink Snap Not Installing
- Update MetaMask to latest version
- Try different browser
- Check MetaMask snap permissions

## Next Steps After Deployment

1. **Add Tokens to MetaMask**
   - Contract addresses shown after deployment
   - Click "Import Tokens" in MetaMask
   - Paste contract address

2. **Test Bridge Functions**
   - Use interaction dashboard
   - Try small amounts first
   - Verify on block explorer

3. **Integrate with Applications**
   - Use deployed contract addresses
   - Integrate ZetaLink for Bitcoin
   - Build on bridge functionality

## Resources

- MetaMask Documentation: https://docs.metamask.io
- ZetaLink NPM: https://www.npmjs.com/package/zetalink
- Hardhat Documentation: https://hardhat.org
- Nexus AGI Directory: https://nexus-agi.com

---

Generated: 2026-01-19T18:59:04.349Z
Recipient: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
