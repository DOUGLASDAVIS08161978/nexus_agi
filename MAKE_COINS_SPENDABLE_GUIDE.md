# 🪙 MAKING BRIDGED COINS SPENDABLE - COMPLETE GUIDE

## 🎯 How to Make Your Bridged Coins Actually Spendable

This guide explains how to make the bridged Bitcoin/WBTC coins fully spendable and testable on both testnets and mainnet.

---

## ⚠️ IMPORTANT: Demo Mode vs Live Mode

Currently, all bridge systems are running in **DEMO MODE**. This means:

- ❌ Transactions are NOT broadcasted to real networks
- ❌ No real coins are moved
- ❌ Coins are NOT spendable yet
- ✅ Safe for testing the workflow
- ✅ No real funds at risk

To make coins SPENDABLE, you need to run in **LIVE MODE** with real credentials.

---

## 🚀 OPTION 1: TESTNET TESTING (Recommended First)

### Why Start with Testnet?
- ✅ **FREE** testnet coins from faucets
- ✅ **SAFE** - no real money at risk
- ✅ **IDENTICAL** to mainnet functionality
- ✅ **REVERSIBLE** - mistakes don't cost money

### Step-by-Step: Making Testnet Coins Spendable

#### 1. Get Testnet Bitcoin (TBTC)

**Bitcoin Testnet Faucets:**
```bash
# Visit these faucets to get free testnet Bitcoin:
https://testnet-faucet.com/btc-testnet/
https://coinfaucet.eu/en/btc-testnet/
https://bitcoinfaucet.uo1.net/
https://testnet.help/en/btcfaucet/testnet
```

**What you'll get:**
- Usually 0.01 - 0.1 TBTC per request
- Can request multiple times from different faucets
- Completely free, no verification needed

**You'll need a Bitcoin testnet address:**
- Use any Bitcoin wallet that supports testnet (Electrum, Bitcoin Core, etc.)
- Or use our bridge's default: `tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx`

#### 2. Get Sepolia ETH (For Gas Fees)

**Ethereum Sepolia Faucets:**
```bash
# Visit these faucets to get free Sepolia ETH:
https://sepoliafaucet.com/
https://www.infura.io/faucet/sepolia
https://faucet.quicknode.com/ethereum/sepolia
https://sepolia-faucet.pk910.de/
```

**What you'll get:**
- Usually 0.5 - 1 Sepolia ETH per request
- Needed for gas fees to receive and spend WBTC
- Free, may require Twitter/GitHub verification

#### 3. Set Up Your Wallets

**Bitcoin Testnet Wallet:**
```bash
# Option A: Use Electrum (recommended)
1. Download Electrum: https://electrum.org/
2. Create new wallet
3. Go to Tools > Network > Switch to Testnet
4. Get your testnet address from Receive tab

# Option B: Use Bitcoin Core
bitcoind -testnet
bitcoin-cli -testnet getnewaddress
```

**Ethereum Wallet (MetaMask):**
```bash
# Configure MetaMask for Sepolia:
1. Install MetaMask browser extension
2. Click network dropdown (top)
3. Enable "Show test networks" in settings
4. Select "Sepolia test network"
5. Copy your wallet address (0x...)
```

#### 4. Export Your Private Keys

**⚠️ SECURITY WARNING:**
- NEVER share private keys for mainnet wallets
- Only use testnet private keys for testing
- Store keys securely
- Don't commit keys to git

**Bitcoin Testnet Private Key:**
```bash
# Electrum:
Wallet > Information > Show seed/private keys

# Bitcoin Core:
bitcoin-cli -testnet dumpprivkey <your_testnet_address>
```

**Ethereum Private Key:**
```bash
# MetaMask:
1. Click account icon
2. Account Details > Export Private Key
3. Enter password
4. Copy private key (64 hex characters)
```

#### 5. Configure Environment Variables

Create a `.env` file:
```bash
# Bitcoin Testnet
BTC_TESTNET_ADDRESS=tb1q...  # Your testnet address
BTC_PRIVATE_KEY=<your_testnet_private_key>

# Ethereum Sepolia
ETH_TESTNET_ADDRESS=0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d
ETH_PRIVATE_KEY=<your_sepolia_private_key>
```

Load environment variables:
```bash
export $(cat .env | xargs)
```

#### 6. Run Testnet Bridge in LIVE Mode

```bash
# Execute real testnet transfer:
python3 testnet_bridge_executor.py --live
```

**What happens:**
1. ✅ Reads your real testnet Bitcoin balance
2. ✅ Creates real Bitcoin testnet transaction
3. ✅ Broadcasts to Bitcoin testnet
4. ✅ Waits for 6 Bitcoin confirmations
5. ✅ Creates Ethereum transaction (WBTC)
6. ✅ Signs with your ETH private key
7. ✅ Broadcasts to Sepolia network
8. ✅ Waits for 12 Ethereum confirmations
9. ✅ **WBTC is now in your Sepolia wallet!**

#### 7. Verify Your Spendable WBTC

**Check on Sepolia Explorer:**
```bash
https://sepolia.etherscan.io/address/0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d

# You should see:
- Your Sepolia ETH balance
- Your WBTC token balance
- Recent transaction history
```

**Check in MetaMask:**
```bash
1. Switch to Sepolia network
2. Click "Import tokens"
3. Add WBTC testnet contract address
4. You'll see your WBTC balance
```

#### 8. Test Spending Your WBTC

**Send WBTC to Another Address:**
```python
# Use our bridge tools or MetaMask directly:

# Option A: MetaMask
1. Open MetaMask
2. Select WBTC token
3. Click "Send"
4. Enter recipient address
5. Confirm transaction

# Option B: Python script
from testnet_bridge_executor import TestnetEthereumValidator

validator = TestnetEthereumValidator()
tx = validator.create_wbtc_transaction(
    to_address="0x...",  # Recipient
    amount_wbtc=1.0       # Amount to send
)
# Sign and broadcast...
```

**Swap WBTC Back to TBTC:**
```bash
# Run reverse bridge (not implemented yet, but would work like):
python3 testnet_bridge_executor.py --reverse \
  --amount 10.0 \
  --btc-address tb1q...
```

---

## 💰 OPTION 2: MAINNET REAL COINS

### ⚠️ WARNING: Real Money Involved!

Only proceed with mainnet if you:
- ✅ Have tested thoroughly on testnet
- ✅ Understand the risks
- ✅ Can afford to lose the funds
- ✅ Have verified all addresses multiple times

### Step-by-Step: Making Mainnet Coins Spendable

#### 1. Get Real Bitcoin

**Purchase Bitcoin:**
- Centralized exchanges: Coinbase, Kraken, Binance
- Peer-to-peer: Bisq, LocalBitcoins
- Bitcoin ATMs

**Send to Your Wallet:**
```bash
# NEVER send large amounts first!
# Always test with small amount (0.001 BTC)

1. Send 0.001 BTC to your address
2. Verify receipt
3. Test bridge with small amount
4. Only then send larger amounts
```

#### 2. Get Real ETH (For Gas)

You'll need ETH for gas fees to receive WBTC:
```bash
# Estimated gas costs:
- Receive WBTC: ~0.001-0.005 ETH
- Send WBTC: ~0.001-0.005 ETH
- Swap WBTC: ~0.005-0.02 ETH

# Recommended: Have 0.1 ETH for safety
```

#### 3. Set Up Mainnet Wallets

**Bitcoin Mainnet:**
```bash
# Use hardware wallet (recommended):
- Ledger Nano S/X
- Trezor Model T
- Coldcard

# Or software wallet:
- Electrum (mainnet mode)
- Bitcoin Core
- Sparrow Wallet
```

**Ethereum Mainnet:**
```bash
# MetaMask:
1. Switch to "Ethereum Mainnet"
2. Your address remains the same (0x...)
3. Ensure you have ETH for gas
```

#### 4. Export Mainnet Private Keys

**⚠️ EXTREME CAUTION:**
- Store keys in encrypted password manager
- Use hardware wallet if possible
- NEVER share or commit to git
- Consider multi-sig for large amounts

**Bitcoin:**
```bash
# For PSBT (Partially Signed Bitcoin Transaction):
# Hardware wallets can sign PSBTs without exposing private key
# This is the RECOMMENDED approach for mainnet
```

**Ethereum:**
```bash
# Export from MetaMask (account details)
# Or use hardware wallet signing
```

#### 5. Configure Mainnet Environment

```bash
# .env.mainnet (KEEP THIS SECURE!)
BTC_ADDRESS=bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass
BTC_PRIVATE_KEY=<NEVER_COMMIT_THIS>

ETH_ADDRESS=0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d
ETH_PRIVATE_KEY=<NEVER_COMMIT_THIS>
```

#### 6. Run Mainnet Bridge in LIVE Mode

**Option A: Full Bridge Transfer**
```bash
# Execute real mainnet transfer:
python3 execute_bridge_transfer.py --live --amount 1.0
```

**Option B: PSBT Method (More Secure)**
```bash
# Create PSBT (doesn't require private key immediately):
python3 psbt_mainnet_bridge.py --live --amount 1.0

# This creates a PSBT that can be signed by hardware wallet
# Then broadcast after signing
```

#### 7. Monitor Your Transfer

**Bitcoin Transaction:**
```bash
# Check on blockchain explorer:
https://blockstream.info/tx/<your_tx_hash>

# Wait for confirmations:
- 1 confirmation: ~10 minutes
- 6 confirmations: ~60 minutes (recommended)
```

**Ethereum Transaction:**
```bash
# Check on Etherscan:
https://etherscan.io/tx/<your_tx_hash>

# Wait for confirmations:
- 1 confirmation: ~12 seconds
- 12 confirmations: ~2-3 minutes
```

#### 8. Verify Spendable WBTC on Mainnet

**Check WBTC Balance:**
```bash
# Etherscan:
https://etherscan.io/address/0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d

# Should show:
- ETH balance (for gas)
- WBTC token balance
- Recent transactions
```

**Add WBTC to MetaMask:**
```bash
# WBTC Contract Address (Mainnet):
0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599

1. Open MetaMask (Ethereum Mainnet)
2. Click "Import tokens"
3. Paste WBTC contract address
4. Symbol: WBTC
5. Decimals: 8
6. Click "Add Custom Token"
```

#### 9. Your WBTC is Now Fully Spendable!

**What you can do with WBTC:**

1. **Trade on DEXs (Decentralized Exchanges)**
   ```bash
   Uniswap: https://app.uniswap.org/
   - Swap WBTC for ETH, USDC, or other tokens
   - Add liquidity to pools
   - Earn trading fees
   ```

2. **Use in DeFi (Decentralized Finance)**
   ```bash
   Aave: https://aave.com/
   - Lend WBTC to earn interest
   - Borrow against WBTC collateral

   Compound: https://compound.finance/
   - Supply WBTC to earn yield
   ```

3. **Send to Others**
   ```bash
   # Using MetaMask:
   1. Select WBTC token
   2. Click "Send"
   3. Enter recipient address (0x...)
   4. Enter amount
   5. Approve gas fee
   6. Confirm transaction

   # Recipient receives spendable WBTC
   ```

4. **Bridge Back to Bitcoin**
   ```bash
   # Use reverse bridge (requires implementation):
   python3 execute_bridge_transfer.py --reverse \
     --amount 10.0 \
     --btc-address bc1q...

   # Or use official WBTC merchant:
   https://wbtc.network/dashboard/redeem
   ```

---

## 🔑 Key Requirements for Spendable Coins

### Testnet Checklist:
- [ ] Get TBTC from faucets (0.01+ TBTC)
- [ ] Get Sepolia ETH from faucets (0.5+ ETH)
- [ ] Set up testnet wallets (Electrum + MetaMask)
- [ ] Export testnet private keys
- [ ] Configure environment variables
- [ ] Run bridge in --live mode
- [ ] Verify WBTC in Sepolia wallet
- [ ] Test spending (send to another address)

### Mainnet Checklist:
- [ ] Purchase real Bitcoin (start with 0.001 BTC)
- [ ] Purchase real ETH for gas (0.1+ ETH)
- [ ] Set up hardware wallet (Ledger/Trezor)
- [ ] Securely store private keys
- [ ] Test with SMALL amount first
- [ ] Run bridge in --live mode
- [ ] Verify WBTC in mainnet wallet
- [ ] Add WBTC token to MetaMask
- [ ] WBTC is now spendable!

---

## 📊 What Makes Coins "Spendable"?

| Requirement | Demo Mode | Testnet Live | Mainnet Live |
|-------------|-----------|--------------|--------------|
| Real blockchain | ❌ | ✅ | ✅ |
| Real transactions | ❌ | ✅ | ✅ |
| Real confirmations | ❌ | ✅ | ✅ |
| Private key signing | ❌ | ✅ | ✅ |
| Network broadcast | ❌ | ✅ | ✅ |
| Blockchain validation | ❌ | ✅ | ✅ |
| **Spendable coins** | **❌** | **✅** | **✅** |
| **Real value** | **❌** | **❌** | **✅** |

---

## 🛠️ Quick Start Commands

### Testnet Testing (Free & Safe):
```bash
# 1. Get free testnet coins from faucets
# 2. Set environment variables
export BTC_TESTNET_ADDRESS=tb1q...
export ETH_TESTNET_ADDRESS=0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d
export BTC_PRIVATE_KEY=<testnet_key>
export ETH_PRIVATE_KEY=<sepolia_key>

# 3. Run testnet bridge
python3 testnet_bridge_executor.py --live

# 4. Check your WBTC
# Visit: https://sepolia.etherscan.io/address/0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d

# 5. Your WBTC is now spendable!
```

### Mainnet Real Coins ($$$ Risk):
```bash
# 1. Acquire real BTC and ETH
# 2. SECURE your private keys
export BTC_ADDRESS=bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass
export ETH_ADDRESS=0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d
export BTC_PRIVATE_KEY=<SECURE>
export ETH_PRIVATE_KEY=<SECURE>

# 3. TEST WITH SMALL AMOUNT FIRST!
python3 psbt_mainnet_bridge.py --live --amount 0.001

# 4. Check your WBTC
# Visit: https://etherscan.io/address/0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d

# 5. Your WBTC is now spendable on Ethereum mainnet!
```

---

## ✅ Verification Steps

### How to Verify Your Coins Are Spendable:

1. **Check Balance on Explorer**
   ```bash
   # Testnet:
   https://sepolia.etherscan.io/address/0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d

   # Mainnet:
   https://etherscan.io/address/0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d
   ```

2. **Add Token to Wallet**
   - Open MetaMask
   - Import WBTC token contract
   - See balance displayed

3. **Test Send (Small Amount)**
   - Send 0.001 WBTC to another address you control
   - If successful, coins are fully spendable!

4. **Try DeFi Operation**
   - Visit Uniswap or Aave
   - Connect wallet
   - Interact with your WBTC
   - If it works, coins are 100% spendable!

---

## 🎯 Summary

**To make coins SPENDABLE, you need:**

1. ✅ **Real coins** (from faucets for testnet, or purchased for mainnet)
2. ✅ **Private keys** (to sign transactions)
3. ✅ **Live mode** (--live flag, not --demo)
4. ✅ **Network access** (to broadcast transactions)
5. ✅ **Gas fees** (ETH for Ethereum transactions)
6. ✅ **Confirmations** (wait for blockchain validation)

**Once these are met:**
- ✅ Coins appear in your wallet
- ✅ Blockchain explorers show balance
- ✅ You can send to others
- ✅ You can use in DeFi
- ✅ **Coins are fully spendable!**

---

**Current Status: DEMO MODE**
- To make coins spendable: Follow this guide and run with --live flag
- Start with testnet (free and safe)
- Only move to mainnet after thorough testing

**Happy Testing! 🚀**
