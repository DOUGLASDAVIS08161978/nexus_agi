# 💰 How to Check Your Bitcoin Wallet Balances

## ⚡ SUPER QUICK METHOD (30 seconds)

**On ANY computer with internet access:**

### Option 1: One-Line Command (Instant!)

```bash
# Copy and paste this ONE command:
curl -s https://blockstream.info/api/address/bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass | python3 -c "import sys,json;d=json.load(sys.stdin);print(f\"PRIMARY Balance: {(d['chain_stats']['funded_txo_sum']-d['chain_stats']['spent_txo_sum'])/100_000_000:.8f} BTC\")"
```

**That's it!** You'll see your PRIMARY wallet balance immediately.

---

### Option 2: Simple Script (All 3 Wallets)

```bash
# Clone repo
git clone https://github.com/DOUGLASDAVIS08161978/nexus_agi
cd nexus_agi

# Run simple checker
./check_balances_simple.sh
```

Output:
```
💰 PRIMARY:   0.XXXXXXXX BTC (XXX,XXX satoshis)
💰 SECONDARY: 0.XXXXXXXX BTC (XXX,XXX satoshis)
💰 LIGHTNING: 0.XXXXXXXX BTC (XXX,XXX satoshis)
```

---

### Option 3: Full Deployment Test (Complete Info)

```bash
# In the repository
./quick_deploy.sh
```

This gives you:
- ✅ All 3 wallet balances
- ✅ Transaction counts
- ✅ Total BTC + USD value
- ✅ Network statistics
- ✅ Full JSON report

---

## 📱 From Your Phone

### iPhone/iPad (Using iSH or similar):
```bash
# Install iSH from App Store
# Open iSH and run:
apk add python3 curl
curl -s https://blockstream.info/api/address/bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass | python3 -c "import sys,json;d=json.load(sys.stdin);print(f\"{(d['chain_stats']['funded_txo_sum']-d['chain_stats']['spent_txo_sum'])/100_000_000:.8f} BTC\")"
```

### Android (Using Termux):
```bash
# Install Termux from F-Droid or Play Store
# Open Termux and run:
pkg install python curl
curl -s https://blockstream.info/api/address/bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass | python3 -c "import sys,json;d=json.load(sys.stdin);print(f\"{(d['chain_stats']['funded_txo_sum']-d['chain_stats']['spent_txo_sum'])/100_000_000:.8f} BTC\")"
```

---

## 🌐 Online (No Install Required)

### Using Online Python:
1. Go to: https://repl.it/languages/python3
2. Paste this code:
```python
import requests
addr = 'bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass'
r = requests.get(f'https://blockstream.info/api/address/{addr}')
data = r.json()
balance = (data['chain_stats']['funded_txo_sum'] - data['chain_stats']['spent_txo_sum']) / 100_000_000
print(f'PRIMARY Balance: {balance:.8f} BTC')
```
3. Click "Run"

---

## 🖥️ By Operating System

### Windows
```powershell
# PowerShell
python -c "import requests; addr='bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass'; r=requests.get(f'https://blockstream.info/api/address/{addr}'); data=r.json(); print(f'Balance: {(data[\"chain_stats\"][\"funded_txo_sum\"]-data[\"chain_stats\"][\"spent_txo_sum\"])/100_000_000:.8f} BTC')"
```

### macOS
```bash
# Terminal
python3 -c "import requests; addr='bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass'; r=requests.get(f'https://blockstream.info/api/address/{addr}'); data=r.json(); print(f'Balance: {(data[\"chain_stats\"][\"funded_txo_sum\"]-data[\"chain_stats\"][\"spent_txo_sum\"])/100_000_000:.8f} BTC')"
```

### Linux
```bash
# Terminal
python3 -c "import requests; addr='bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass'; r=requests.get(f'https://blockstream.info/api/address/{addr}'); data=r.json(); print(f'Balance: {(data['chain_stats']['funded_txo_sum']-data['chain_stats']['spent_txo_sum'])/100_000_000:.8f} BTC')"
```

---

## 💻 Your Wallet Addresses

```
PRIMARY:   bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass
SECONDARY: bc1q8z6z78dy5squapjpkeruem98jcezsw37hnae6qjyhxma6jmxyn6qsmqxce
LIGHTNING: bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
```

**View online:**
- https://blockstream.info/address/bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass
- https://mempool.space/address/bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass

---

## 🔐 Security Note

These are **watch-only addresses** (public addresses only).
- ✅ Safe to check publicly
- ✅ No private keys involved
- ✅ Cannot spend funds
- ✅ Only reads public blockchain data

---

## 🆘 Troubleshooting

### "requests module not found"
```bash
pip3 install requests
```

### "curl command not found"
```bash
# Ubuntu/Debian
sudo apt-get install curl

# macOS
brew install curl

# CentOS/RHEL
sudo yum install curl
```

### Network Timeout
Try alternative API:
```bash
curl -s https://mempool.space/api/address/bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass/utxo | python3 -c "import sys,json;utxos=json.load(sys.stdin);print(f'Balance: {sum(u[\"value\"] for u in utxos)/100_000_000:.8f} BTC')"
```

---

## 📊 What the Numbers Mean

- **Satoshis**: Smallest unit of Bitcoin (1 BTC = 100,000,000 satoshis)
- **BTC**: Bitcoin in decimal format
- **Transactions**: Number of times the address has been used

---

## ⚡ TL;DR - Fastest Method

**Copy this. Paste in terminal. Press enter:**

```bash
curl -s https://blockstream.info/api/address/bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass | python3 -c "import sys,json;d=json.load(sys.stdin);print(f\"{(d['chain_stats']['funded_txo_sum']-d['chain_stats']['spent_txo_sum'])/100_000_000:.8f} BTC\")"
```

**Done!** 🚀

---

*For complete deployment with all features, see: DEPLOYMENT_GUIDE.md*
