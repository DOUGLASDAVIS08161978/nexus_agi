# 🔐 EXTERNAL SIGNING GUIDE - Hardware Wallets & Other Tools

## 🎯 Signing Transactions with External Tools

This guide shows you how to sign Bitcoin PSBTs and Ethereum transactions using external tools like hardware wallets, without exposing your private keys to the bridge software.

---

## ⚡ WHY USE EXTERNAL SIGNING?

### Security Benefits:
- ✅ **Private keys NEVER leave hardware wallet**
- ✅ **Immune to malware on computer**
- ✅ **Physical confirmation required**
- ✅ **Industry best practice for large amounts**
- ✅ **Supports multi-signature**

### Recommended For:
- Mainnet transfers (real money)
- Large amounts (> 1 BTC)
- Long-term storage
- Business/institutional use

---

## 🪙 PART 1: SIGNING BITCOIN PSBTs

### What is PSBT?
**PSBT** = Partially Signed Bitcoin Transaction (BIP 174)
- Industry-standard format for unsigned Bitcoin transactions
- Can be created offline
- Signed by hardware wallets
- Compatible with all major tools

### Method 1: Hardware Wallet (Ledger)

#### Step 1: Create PSBT (Without Private Key)
```bash
# Our bridge creates PSBT automatically:
python3 psbt_mainnet_bridge.py --demo --amount 1.0

# This creates:
# - psbt_mainnet_records.json (with PSBT data)
# - PSBT remains unsigned
```

#### Step 2: Export PSBT
```bash
# PSBT is saved in base64 format
cat psbt_mainnet_records.json | jq -r '.[0].psbt.psbt_base64' > unsigned.psbt

# View PSBT details:
bitcoin-cli decodepsbt $(cat unsigned.psbt)
```

#### Step 3: Sign with Ledger Hardware Wallet
```bash
# Using Bitcoin Core:
bitcoin-cli walletprocesspsbt $(cat unsigned.psbt)

# Using Ledger Live:
1. Connect Ledger device
2. Open Bitcoin app on Ledger
3. Import PSBT file
4. Review transaction on device
5. Physically press button to sign
6. Export signed PSBT
```

#### Step 4: Broadcast Signed PSBT
```bash
# Extract signed transaction:
bitcoin-cli finalizepsbt $(cat signed.psbt)

# Broadcast to network:
bitcoin-cli sendrawtransaction <hex_transaction>

# Or use our bridge to continue:
python3 psbt_mainnet_bridge.py --resume --signed-psbt signed.psbt
```

### Method 2: Hardware Wallet (Trezor)

#### Using Trezor Suite:
```bash
1. Open Trezor Suite
2. Go to Accounts > Bitcoin
3. Click "Send"
4. Import PSBT file
5. Review on Trezor screen
6. Confirm with physical button
7. Export signed PSBT
```

#### Using Electrum with Trezor:
```bash
1. Open Electrum connected to Trezor
2. Tools > Load transaction > From file
3. Select unsigned.psbt
4. Click "Sign"
5. Confirm on Trezor device
6. Export signed transaction
```

### Method 3: Hardware Wallet (Coldcard)

```bash
# Coldcard excels at PSBT signing:

1. Create PSBT with our bridge
2. Save PSBT to MicroSD card
3. Insert MicroSD into Coldcard
4. On Coldcard: Ready to Sign > select PSBT
5. Review transaction details
6. Confirm with PIN
7. Signed PSBT saved to MicroSD
8. Load signed PSBT and broadcast
```

### Method 4: Sparrow Wallet

```bash
# Sparrow Wallet is excellent for PSBT workflow:

1. Open Sparrow Wallet
2. File > Open Transaction > From File
3. Select unsigned.psbt
4. Click "Sign"
   - With hardware wallet, or
   - With hot wallet
5. View signed transaction
6. Click "Broadcast" to send
```

### Method 5: Bitcoin Core CLI

```bash
# Full workflow with Bitcoin Core:

# 1. Create PSBT (alternative to our bridge)
bitcoin-cli walletcreatefundedpsbt \
  '[{"txid":"<utxo_txid>","vout":0}]' \
  '[{"<destination_address>":0.1}]' \
  0 \
  '{"includeWatching":true,"changeAddress":"<your_address>"}'

# 2. Process (sign) PSBT
bitcoin-cli walletprocesspsbt "<psbt_base64>" true

# 3. Finalize PSBT
bitcoin-cli finalizepsbt "<signed_psbt_base64>"

# 4. Broadcast
bitcoin-cli sendrawtransaction "<hex_transaction>"
```

---

## ⚡ PART 2: SIGNING ETHEREUM TRANSACTIONS

### Method 1: MetaMask (Browser Extension)

#### Step 1: Create Unsigned Transaction
```python
# Our bridge creates unsigned transaction:
from psbt_mainnet_bridge import EthereumMainnetBroadcaster

broadcaster = EthereumMainnetBroadcaster()
tx = broadcaster.create_wbtc_transaction(
    to_address="0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d",
    amount_wbtc=299.7
)

# Saves as JSON: unsigned_eth_tx.json
import json
with open('unsigned_eth_tx.json', 'w') as f:
    json.dump(tx, f, indent=2)
```

#### Step 2: Sign with MetaMask
```javascript
// In browser console (with MetaMask installed):
const tx = {
  nonce: "0x...",
  to: "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", // WBTC
  value: "0x0",
  gas: "0x249F0",
  gasPrice: "0x...",
  chainId: 1,
  data: "0x..."
};

// Request signature from MetaMask:
const signedTx = await ethereum.request({
  method: 'eth_signTransaction',
  params: [tx]
});

console.log("Signed TX:", signedTx);
```

#### Step 3: Broadcast
```javascript
// Broadcast signed transaction:
const txHash = await ethereum.request({
  method: 'eth_sendRawTransaction',
  params: [signedTx]
});

console.log("TX Hash:", txHash);
console.log("Etherscan:", `https://etherscan.io/tx/${txHash}`);
```

### Method 2: Ledger Ethereum App

```bash
# Using Ledger with Ethereum:

1. Connect Ledger device
2. Open Ethereum app on Ledger
3. Use tool to send transaction:

# With ledgerctl:
ledgerctl sign-tx \
  --to 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599 \
  --value 0 \
  --data 0xa9059cbb... \
  --gas-limit 150000 \
  --gas-price 30000000000

# Physically confirm on Ledger
# Returns signed transaction
```

### Method 3: MyEtherWallet (MEW) with Hardware Wallet

```bash
1. Go to https://www.myetherwallet.com/
2. Click "Access Wallet"
3. Select "Hardware" > "Ledger" or "Trezor"
4. Connect device and unlock
5. Select Ethereum account
6. Click "Send Transaction"
7. Enter:
   - To: 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599 (WBTC)
   - Amount: 0 ETH
   - Data: <our bridge transaction data>
8. Confirm on hardware device
9. Transaction broadcasted
```

### Method 4: Web3.py with Hardware Wallet

```python
from web3.auto import w3
from ledgerblue.comm import getDongle

# Connect to Ledger
dongle = getDongle(True)

# Create transaction
tx = {
    'nonce': w3.eth.get_transaction_count('0x...'),
    'to': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',
    'value': 0,
    'gas': 150000,
    'gasPrice': w3.eth.gas_price,
    'chainId': 1,
    'data': '0xa9059cbb...'
}

# Sign with Ledger (requires ledgerblue package)
signed_tx = w3.eth.account.sign_transaction(tx, dongle)

# Broadcast
tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
print(f"TX: {tx_hash.hex()}")
```

### Method 5: Ethers.js with Ledger

```javascript
const { ethers } = require("ethers");
const { LedgerSigner } = require("@ethersproject/hardware-wallets");

async function signWithLedger() {
  // Connect to Ledger
  const ledger = new LedgerSigner(provider, "m/44'/60'/0'/0/0");

  // Create transaction
  const tx = {
    to: "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    value: 0,
    gasLimit: 150000,
    gasPrice: ethers.utils.parseUnits("30", "gwei"),
    data: "0xa9059cbb..." // WBTC transfer data
  };

  // Sign with Ledger (physical confirmation required)
  const signedTx = await ledger.signTransaction(tx);

  // Broadcast
  const txResponse = await provider.sendTransaction(signedTx);
  console.log("TX Hash:", txResponse.hash);
}
```

---

## 🔧 PART 3: COMPLETE BRIDGE WORKFLOW WITH EXTERNAL SIGNING

### Bitcoin → Ethereum Bridge with Hardware Wallets

```bash
# STEP 1: Create unsigned PSBT (Bitcoin side)
python3 psbt_mainnet_bridge.py --create-psbt-only --amount 1.0
# Output: unsigned_bitcoin.psbt

# STEP 2: Sign Bitcoin PSBT with Hardware Wallet
# Using Ledger:
bitcoin-cli walletprocesspsbt $(cat unsigned_bitcoin.psbt)
# Or: Use Ledger Live, Trezor Suite, Sparrow, etc.
# Output: signed_bitcoin.psbt

# STEP 3: Broadcast Bitcoin transaction
bitcoin-cli finalizepsbt $(cat signed_bitcoin.psbt)
bitcoin-cli sendrawtransaction <hex>
# Get transaction hash

# STEP 4: Wait for Bitcoin confirmations (6 blocks)
# Monitor: https://blockstream.info/tx/<hash>

# STEP 5: Create unsigned Ethereum transaction
python3 psbt_mainnet_bridge.py \
  --create-eth-tx \
  --btc-tx <bitcoin_tx_hash> \
  --amount 0.999  # (1.0 BTC minus 0.1% fee)
# Output: unsigned_ethereum.json

# STEP 6: Sign Ethereum transaction with Hardware Wallet
# Using Ledger via MyEtherWallet or Ledger Live
# Or: Use MetaMask, Trezor Suite, etc.
# Output: signed_ethereum.json

# STEP 7: Broadcast Ethereum transaction
python3 psbt_mainnet_bridge.py \
  --broadcast-eth \
  --signed-tx signed_ethereum.json
# Output: Ethereum TX hash

# STEP 8: Wait for Ethereum confirmations (12 blocks)
# Monitor: https://etherscan.io/tx/<hash>

# STEP 9: Verify WBTC in wallet
# Check: https://etherscan.io/address/0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d

# ✅ DONE! Your WBTC is spendable!
```

---

## 🛡️ SECURITY BEST PRACTICES

### For Hardware Wallets:

1. **Always Verify on Device Screen**
   - Check destination address
   - Check amount
   - Check fee
   - Never approve blindly

2. **Use Latest Firmware**
   - Keep hardware wallet updated
   - Use official apps only
   - Verify firmware signatures

3. **Test with Small Amounts First**
   - Send 0.001 BTC first
   - Verify receipt
   - Then send larger amounts

4. **Double-Check Addresses**
   - Write down address
   - Compare character by character
   - One typo = lost funds forever

5. **Secure Backup**
   - Store seed phrase offline
   - Metal backup recommended
   - Never digital photos
   - Multiple secure locations

### For Software Signing:

1. **Use Air-Gapped Computer**
   - Never connected to internet
   - Sign transactions offline
   - Transfer via QR code or USB

2. **Verify Transaction Data**
   - Decode PSBT before signing
   - Check all inputs/outputs
   - Verify amounts and addresses

3. **Use Multisig for Large Amounts**
   - Require 2-of-3 or 3-of-5 signatures
   - Different devices/locations
   - No single point of failure

---

## 📋 RECOMMENDED TOOLS

### Bitcoin PSBT Signing:

| Tool | Type | Best For | Difficulty |
|------|------|----------|------------|
| **Ledger** | Hardware | Maximum security | Easy |
| **Trezor** | Hardware | User-friendly | Easy |
| **Coldcard** | Hardware | Bitcoin-only, experts | Medium |
| **Sparrow** | Software | Desktop PSBT workflow | Easy |
| **Electrum** | Software | Advanced users | Medium |
| **Bitcoin Core** | Software | Full node operators | Hard |

### Ethereum Transaction Signing:

| Tool | Type | Best For | Difficulty |
|------|------|----------|------------|
| **Ledger** | Hardware | Maximum security | Easy |
| **Trezor** | Hardware | User-friendly | Easy |
| **MetaMask** | Software | Browser convenience | Easy |
| **MyEtherWallet** | Software | Hardware wallet support | Easy |
| **Frame** | Software | Desktop signing | Medium |
| **Gnosis Safe** | Software | Multisig | Medium |

---

## ⚡ QUICK START: TESTNET WITH EXTERNAL SIGNING

### Bitcoin Testnet + Ledger:

```bash
# 1. Get testnet Bitcoin from faucet
# 2. Create PSBT (testnet mode):
python3 testnet_bridge_executor.py --create-psbt --amount 0.1

# 3. Sign with Ledger (in testnet mode):
bitcoin-cli -testnet walletprocesspsbt $(cat unsigned.psbt)

# 4. Continue bridge process:
python3 testnet_bridge_executor.py --signed-psbt signed.psbt --live

# ✅ Testnet WBTC received on Sepolia!
```

### Ethereum Sepolia + MetaMask:

```bash
# 1. Get Sepolia ETH from faucet
# 2. Create unsigned transaction:
python3 testnet_bridge_executor.py --create-eth-tx

# 3. Sign with MetaMask:
# - Import transaction into MetaMask
# - Review and confirm
# - MetaMask signs automatically

# 4. Broadcast:
python3 testnet_bridge_executor.py --broadcast-eth --signed-tx signed.json

# ✅ WBTC transferred on Sepolia!
```

---

## 🎯 FULL EXAMPLE: MAINNET TRANSFER WITH LEDGER

```bash
# Complete workflow using Ledger for both Bitcoin and Ethereum:

# ========================================
# BITCOIN SIDE (Create & Sign PSBT)
# ========================================

# 1. Create PSBT
python3 psbt_mainnet_bridge.py --create-psbt-only \
  --btc-address bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass \
  --amount 1.0

# Output:
# ✓ PSBT created: psbt_abc123.psbt
# ✓ PSBT base64: cHNidP8BAH...

# 2. Verify PSBT (before signing!)
bitcoin-cli decodepsbt $(cat psbt_abc123.psbt)

# Check output:
# - Inputs: Correct UTXO?
# - Outputs: Correct destination?
# - Amount: 1.0 BTC?
# - Fee: Reasonable? (~0.00002 BTC)

# 3. Connect Ledger and sign
# Ledger Nano S/X:
# - Connect USB
# - Enter PIN
# - Open Bitcoin app

# Sign with Bitcoin Core + Ledger:
bitcoin-cli walletprocesspsbt $(cat psbt_abc123.psbt) true

# Or sign with Ledger Live:
# - Import PSBT
# - Review on device
# - Press both buttons to confirm
# - Export signed PSBT

# Output: signed_btc.psbt

# 4. Finalize and broadcast Bitcoin TX
bitcoin-cli finalizepsbt $(cat signed_btc.psbt) true

# Output: {
#   "hex": "0200000001...",
#   "complete": true
# }

bitcoin-cli sendrawtransaction <hex_from_above>

# Output: <bitcoin_tx_hash>
# Example: a7b8c9d0e1f2g3h4i5j6k7l8m9n0o1p2q3r4s5t6u7v8w9x0y1z2

# 5. Monitor Bitcoin confirmations
watch -n 10 'bitcoin-cli gettransaction <bitcoin_tx_hash>'

# Wait for 6 confirmations (~60 minutes)

# ========================================
# ETHEREUM SIDE (Create & Sign TX)
# ========================================

# 6. Create Ethereum WBTC transaction
python3 psbt_mainnet_bridge.py --create-eth-tx \
  --btc-tx-hash <bitcoin_tx_hash> \
  --eth-address 0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d \
  --amount 0.999  # 1.0 BTC minus 0.1% bridge fee

# Output: unsigned_eth_tx.json
# Contains:
# - To: 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599 (WBTC contract)
# - Data: ERC20 transfer function call
# - Gas: 150000
# - Gas price: current market rate + 20%

# 7. Sign with Ledger Ethereum app
# Ledger Nano S/X:
# - Close Bitcoin app
# - Open Ethereum app
# - Enable "Contract data" and "Debug data" in settings

# Option A: Sign via MyEtherWallet
# 1. Go to https://www.myetherwallet.com/
# 2. Access Wallet > Hardware > Ledger
# 3. Send Transaction > import unsigned_eth_tx.json
# 4. Review on Ledger screen
# 5. Press both buttons to confirm
# 6. Get signed transaction

# Option B: Sign via Ledgerctl (command line)
ledgerctl sign-eth-tx \
  --tx-file unsigned_eth_tx.json \
  --output signed_eth_tx.json

# Ledger displays:
# - "Contract Data"
# - "To: 0x2260FAC5..."
# - "Amount: 0 ETH"
# - Confirm? [Press both buttons]

# Output: signed_eth_tx.json

# 8. Broadcast Ethereum transaction
python3 psbt_mainnet_bridge.py --broadcast-eth \
  --signed-tx signed_eth_tx.json

# Output:
# ✓ Broadcasting to Ethereum mainnet...
# ✓ TX Hash: 0xef19d8f8ce9c4fe831635f98c940f786a848c4d0...
# ✓ Etherscan: https://etherscan.io/tx/0xef19d8...
# ✓ Waiting for confirmations...
# ✓ Confirmation 1/12
# ✓ Confirmation 2/12
# ...
# ✓ Confirmation 12/12
# ✅ Transaction confirmed!

# 9. Verify WBTC in wallet
# Etherscan:
https://etherscan.io/address/0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d

# Should show:
# - Token: WBTC
# - Balance: 0.999 WBTC (or whatever amount minus bridge fee)

# 10. Add WBTC to MetaMask
# Contract: 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599
# Symbol: WBTC
# Decimals: 8

# ✅ COMPLETE! Your WBTC is now spendable on Ethereum!
```

---

## 🎉 SUCCESS! Your Coins Are Spendable!

Once you've completed the external signing workflow:

✅ **Private keys never exposed** to bridge software
✅ **Physical confirmation** on hardware device
✅ **Maximum security** for your funds
✅ **Fully spendable** WBTC on Ethereum
✅ **Ready for DeFi**, trading, or sending to others

---

## 📚 Additional Resources

### Documentation:
- **BIP 174 (PSBT):** https://github.com/bitcoin/bips/blob/master/bip-0174.mediawiki
- **Ledger Dev Docs:** https://developers.ledger.com/
- **Trezor Dev Docs:** https://docs.trezor.io/
- **EIP-155 (Ethereum TX):** https://eips.ethereum.org/EIPS/eip-155

### Tools:
- **Bitcoin Core:** https://bitcoin.org/en/download
- **Electrum:** https://electrum.org/
- **Sparrow Wallet:** https://sparrowwallet.com/
- **MyEtherWallet:** https://www.myetherwallet.com/
- **Frame:** https://frame.sh/

### Support:
- **Ledger Support:** https://support.ledger.com/
- **Trezor Support:** https://trezor.io/support/
- **WBTC FAQ:** https://wbtc.network/faq

---

**Happy Signing! 🔐**

*Remember: Never share your private keys or seed phrases with anyone, including this software. Hardware wallets keep your keys secure!*
