# Bitcoin Testnet Wallet - Complete Setup Guide

## Your Testnet Wallet

✅ **Wallet Created**: `nexus_testnet_wallet`
✅ **Network**: Bitcoin Testnet (safe for learning, no real money)
✅ **Address Type**: SegWit (tb1... format)

---

## 🎯 Quick Start Guide

### Step 1: Get Your Receiving Address

```bash
python3 working_testnet_wallet.py address
```

This will show your testnet address where you can receive FREE testnet Bitcoin.

### Step 2: Get FREE Testnet Coins

Visit any of these faucets and enter your testnet address:

1. **Mempool Testnet Faucet** (Recommended)
   - https://testnet-faucet.mempool.co/
   - Gives ~0.001 tBTC instantly
   - No registration required

2. **CoinFaucet**
   - https://coinfaucet.eu/en/btc-testnet/
   - Gives ~0.01 tBTC
   - Simple captcha

3. **Bitcoin Faucet**
   - https://bitcoinfaucet.uo1.net/
   - Various amounts available

**Wait Time**: 10-30 minutes for first confirmation

### Step 3: Check Your Balance

```bash
python3 working_testnet_wallet.py balance
```

This scans the blockchain and shows:
- Total balance
- Number of UTXOs
- Recent transactions

### Step 4: Send a Transaction

```bash
python3 working_testnet_wallet.py
```

Then select option 4 from the menu:
- Enter destination TESTNET address (must start with tb1, n, m, or 2)
- Enter amount in satoshis
- Review transaction details
- Confirm to broadcast

---

## 📋 Available Commands

### Interactive Menu
```bash
python3 working_testnet_wallet.py
```

Options:
1. Create/Load Wallet
2. Get Receiving Address
3. Check Balance
4. Send Transaction
5. View Transaction History
6. Export Wallet Info
7. Exit

### Quick Commands
```bash
python3 working_testnet_wallet.py address   # Get receiving address
python3 working_testnet_wallet.py balance   # Check balance
python3 working_testnet_wallet.py history   # View transactions
```

---

## 🔐 Security Features

✅ **Testnet Only** - Script validates all addresses are testnet format
✅ **Balance Checks** - Won't create transaction if insufficient funds
✅ **Confirmation Required** - Asks before broadcasting transactions
✅ **Transaction Logging** - Saves all transaction details to JSON files
✅ **Wallet Backup** - Can export wallet info anytime

---

## 💡 Common Operations

### Sending Testnet Bitcoin

**IMPORTANT**: Destination address MUST be testnet format!

✅ **Valid Testnet Addresses:**
- `tb1q...` (SegWit, recommended)
- `n...` (Legacy testnet)
- `m...` (Legacy testnet)
- `2...` (P2SH testnet)

❌ **Invalid (Mainnet Addresses):**
- `bc1q...` (Mainnet SegWit)
- `1...` (Mainnet legacy)
- `3...` (Mainnet P2SH)

**Example Send:**
```python
# Correct - Testnet address
To: tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx
Amount: 50000 satoshis (0.0005 tBTC)

# Wrong - Mainnet address (will be rejected)
To: bc1qyhkq7usdfhhhynkjksdqfx32u3rmv94y0htsal
ERROR: Mainnet address cannot be used on testnet!
```

### Converting Your Mainnet Address to Testnet

Your address: `bc1qyhkq7usdfhhhynkjksdqfx32u3rmv94y0htsal` is a **mainnet** address.

To use testnet:
1. Get a testnet wallet (Electrum, Bitcoin Core with -testnet flag, or use our wallet)
2. Generate a testnet address (starts with tb1)
3. Use that for all testnet operations

**Our generated testnet address**: See output from `python3 working_testnet_wallet.py address`

---

## 📊 Understanding Transactions

### Transaction Flow

1. **Create Transaction**
   - Select UTXOs to spend
   - Calculate fees
   - Build transaction structure

2. **Sign Transaction**
   - Use private key to sign inputs
   - Generate witness data (for SegWit)

3. **Broadcast Transaction**
   - Send to testnet network
   - Wait for confirmation (~10 minutes)

4. **Confirmation**
   - Transaction included in block
   - Becomes part of blockchain
   - Can be viewed on block explorers

### Fees

Recommended fee rates:
- **Low priority**: 1,000 sats/kB (~10-30 min)
- **Medium priority**: 5,000 sats/kB (~10 min, default)
- **High priority**: 10,000 sats/kB (~next block)

Testnet fees are low since coins have no value!

---

## 🔍 Verifying Transactions

### On Block Explorers

**Blockstream (Recommended)**
```
https://blockstream.info/testnet/address/YOUR_ADDRESS
https://blockstream.info/testnet/tx/YOUR_TXID
```

**Mempool.space**
```
https://mempool.space/testnet/address/YOUR_ADDRESS
https://mempool.space/testnet/tx/YOUR_TXID
```

### What to Check

✅ **Before Sending:**
- Address is correct (copy-paste, don't type)
- Address is testnet format
- Amount is correct
- You have sufficient balance
- Fee is reasonable

✅ **After Sending:**
- Transaction appears on block explorer
- Status shows "Unconfirmed" or "Pending"
- After ~10 min, shows confirmations
- Destination address received the amount

---

## 🎓 Educational Notes

### Mining vs Validation vs Transactions

**Mining** = Creating NEW blocks
- Requires massive computational power
- Competes with other miners
- First to solve proof-of-work wins
- Receives block reward + transaction fees

**Validation** = Verifying EXISTING blocks
- Checks blocks are legitimate
- Ensures consensus rules followed
- No rewards (just confirms blockchain state)

**Transactions** = Moving Bitcoin between addresses
- Pays small fee to miners
- Included in next block
- Becomes permanent part of blockchain

### Testnet vs Mainnet

| Feature | Testnet | Mainnet |
|---------|---------|---------|
| Address prefix | tb1, n, m, 2 | bc1, 1, 3 |
| Coin value | $0 (worthless) | Real money |
| Purpose | Learning, testing | Real transactions |
| Faucets | Free coins available | No free coins |
| Block time | ~10 minutes | ~10 minutes |
| Security | Lower difficulty | High difficulty |

**You CANNOT send testnet coins to mainnet addresses or vice versa!**

---

## 📝 Example Session

```bash
# 1. Get receiving address
$ python3 working_testnet_wallet.py address
💰 TESTNET RECEIVING ADDRESS:
   tb1qkvyjam64dmkvjav5dkhpfyqpjxq7hqj80u6e6f

# 2. Visit faucet, request coins, wait 10-30 minutes

# 3. Check balance
$ python3 working_testnet_wallet.py balance
💰 WALLET BALANCE:
   0.00100000 tBTC
   (100,000 satoshis)
📊 Unspent Outputs (UTXOs): 1

# 4. Send transaction
$ python3 working_testnet_wallet.py
Select option: 4
Enter TESTNET address: tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx
Enter amount in satoshis: 50000

🔨 CREATING TRANSACTION:
   To: tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx
   Amount: 50,000 sats (0.00050000 tBTC)
   Fee rate: 5,000 sats/kB

✅ TRANSACTION CREATED:
   TXID: abc123...
   Size: 141 bytes
   Fee: 705 sats

Broadcast this transaction? (yes/no): yes

✅ TRANSACTION BROADCAST SUCCESSFUL!
   View on explorer:
   https://blockstream.info/testnet/tx/abc123...
```

---

## 🚀 Advanced Features

### Export Wallet Info
```bash
python3 working_testnet_wallet.py
# Select option 6
```

Saves JSON file with:
- All addresses
- Balance
- UTXO count
- Wallet metadata

### Transaction History
```bash
python3 working_testnet_wallet.py history
```

Shows:
- All transactions (sent and received)
- Amounts
- Confirmation status
- Timestamps

### Programmatic Access

You can also use the wallet in Python scripts:

```python
from working_testnet_wallet import TestnetWalletManager

# Create manager
manager = TestnetWalletManager()

# Load wallet
manager.create_or_load_wallet()

# Check balance
balance_info = manager.check_balance()
print(f"Balance: {balance_info['balance_btc']} tBTC")

# Send transaction
manager.send_transaction(
    to_address="tb1q...",
    amount_sats=50000
)
```

---

## ❓ Troubleshooting

### No UTXOs Found
- Wait longer (confirmations take ~10 minutes)
- Check faucet gave you coins
- Verify address on block explorer

### Transaction Failed
- Check you have sufficient balance
- Ensure destination is testnet address
- Try higher fee rate
- Make sure wallet is scanned

### Can't Send to Mainnet Address
- **This is correct behavior!** Testnet and mainnet are separate
- Convert to testnet address or get new testnet wallet
- Our address `bc1qyhkq7usdfhhhynkjksdqfx32u3rmv94y0htsal` won't work on testnet

---

## 📚 Resources

**Bitcoin Testnet Explorers:**
- https://blockstream.info/testnet/
- https://mempool.space/testnet/
- https://live.blockcypher.com/btc-testnet/

**Testnet Faucets:**
- https://testnet-faucet.mempool.co/
- https://coinfaucet.eu/en/btc-testnet/
- https://bitcoinfaucet.uo1.net/

**Learning Resources:**
- Bitcoin Developer Guide: https://developer.bitcoin.org/
- Mastering Bitcoin (free ebook): https://github.com/bitcoinbook/bitcoinbook
- Bitcoinlib Documentation: https://bitcoinlib.readthedocs.io/

---

## ✅ Next Steps

1. **Get testnet coins** - Visit a faucet with your receiving address
2. **Wait for confirmation** - Check balance after 10-30 minutes
3. **Practice sending** - Send small amounts to test addresses
4. **Learn by doing** - Experiment with different amounts and fees
5. **Explore blockchain** - View your transactions on block explorers

Remember: **Testnet coins have ZERO real-world value** - perfect for learning!
