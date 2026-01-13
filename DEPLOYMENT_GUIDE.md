# NEXUS AGI - Deployment Guide for Unrestricted Network

## 🚀 Quick Deploy: Check Wallet Balances NOW

### Option 1: Direct Python Execution (Fastest)

```bash
# On your unrestricted network (local machine, VPS, cloud server)
cd /path/to/nexus_agi

# Install dependencies
pip3 install requests numpy networkx bitcoinlib web3

# Run deployment test (checks balances automatically)
python3 deploy_unrestricted_network.py
```

**This will:**
- ✅ Check all 3 configured wallet balances on MAINNET
- ✅ Test MAINNET, TESTNET, and SIGNET connectivity
- ✅ Test infinite domain network
- ✅ Generate detailed JSON report
- ✅ Display total BTC balance

---

### Option 2: Docker Deployment (Isolated)

```bash
# Build Docker image
docker build -f Dockerfile.bitcoin -t nexus-bitcoin:latest .

# Run deployment test
docker run --rm nexus-bitcoin:latest

# Or run wallet balance check only
docker run --rm nexus-bitcoin:latest python3 -c "
from bitcoin_network_connector import BitcoinNetworkConnector, BitcoinNetwork
connector = BitcoinNetworkConnector(BitcoinNetwork.MAINNET)
connector.check_configured_wallets()
"
```

---

### Option 3: Docker Compose (Full Stack)

```bash
# Start all services
docker-compose -f docker-compose.bitcoin.yml up -d

# Check logs for wallet balances
docker-compose -f docker-compose.bitcoin.yml logs nexus-bitcoin-mainnet

# View continuous wallet monitoring
docker-compose -f docker-compose.bitcoin.yml logs -f nexus-wallet-monitor

# Stop services
docker-compose -f docker-compose.bitcoin.yml down
```

---

## 📊 What Gets Checked

### Configured Wallets

1. **PRIMARY WALLET**
   - Address: `bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass`
   - Usage: Mining rewards, primary operations

2. **SECONDARY WALLET**
   - Address: `bc1q8z6z78dy5squapjpkeruem98jcezsw37hnae6qjyhxma6jmxyn6qsmqxce`
   - Usage: Secondary operations, backup

3. **LIGHTNING WALLET**
   - Address: `bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh`
   - Usage: Lightning Network operations

---

## 🌐 Deployment Environments

### Local Development Machine

**Requirements:**
- Python 3.8+
- Internet access (no proxy restrictions)
- 100MB free disk space

**Setup:**
```bash
git clone <repository-url>
cd nexus_agi
pip3 install -r requirements.txt
python3 deploy_unrestricted_network.py
```

---

### AWS EC2

**Launch Instance:**
```bash
# Amazon Linux 2 or Ubuntu 22.04
# t2.micro is sufficient

# Connect via SSH
ssh -i key.pem ec2-user@<public-ip>

# Install dependencies
sudo yum install python3 python3-pip git -y  # Amazon Linux
# OR
sudo apt-get install python3 python3-pip git -y  # Ubuntu

# Clone and deploy
git clone <repository-url>
cd nexus_agi
pip3 install requests numpy networkx bitcoinlib web3
python3 deploy_unrestricted_network.py
```

---

### Google Cloud Platform (GCP)

**Cloud Run:**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/nexus-bitcoin
gcloud run deploy nexus-bitcoin \
  --image gcr.io/PROJECT_ID/nexus-bitcoin \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**Compute Engine:**
```bash
# Create VM
gcloud compute instances create nexus-bitcoin \
  --machine-type e2-micro \
  --image-family ubuntu-2204-lts \
  --image-project ubuntu-os-cloud

# SSH and deploy
gcloud compute ssh nexus-bitcoin
git clone <repository-url>
cd nexus_agi
python3 deploy_unrestricted_network.py
```

---

### Azure

**App Service:**
```bash
az webapp up \
  --name nexus-bitcoin \
  --resource-group nexus-rg \
  --runtime "PYTHON:3.11"
```

**Container Instances:**
```bash
az container create \
  --resource-group nexus-rg \
  --name nexus-bitcoin \
  --image nexus-bitcoin:latest \
  --cpu 1 \
  --memory 1
```

---

### DigitalOcean Droplet

```bash
# Create $5/month droplet (Ubuntu 22.04)
# SSH into droplet

apt-get update
apt-get install python3 python3-pip git -y
git clone <repository-url>
cd nexus_agi
pip3 install requests numpy networkx bitcoinlib web3
python3 deploy_unrestricted_network.py
```

---

### Heroku

```bash
# Create app
heroku create nexus-bitcoin

# Add Python buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Run deployment test
heroku run python3 deploy_unrestricted_network.py
```

---

## 📋 Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 512MB minimum, 1GB recommended
- **Disk**: 100MB for dependencies
- **Network**: Unrestricted HTTPS access

### Python Dependencies

```txt
requests>=2.31.0
numpy>=1.24.0
networkx>=3.0
bitcoinlib>=0.6.14
web3>=6.0.0
```

Install with:
```bash
pip3 install requests numpy networkx bitcoinlib web3
```

---

## 🔐 Security Best Practices

### Network Security

1. **Firewall Configuration**
   ```bash
   # Allow only necessary ports
   sudo ufw allow 22/tcp    # SSH
   sudo ufw allow 8000/tcp  # API (if needed)
   sudo ufw enable
   ```

2. **API Rate Limiting**
   - Built-in automatic failover between multiple APIs
   - Respects rate limits automatically

3. **Wallet Security**
   - **IMPORTANT**: These are WATCH-ONLY addresses
   - No private keys in the code
   - Cannot spend funds, only check balances
   - Safe to run on any server

### Private Key Management

**NEVER store private keys in:**
- ❌ Source code
- ❌ Environment variables (on shared systems)
- ❌ Database (unless encrypted)
- ❌ Log files

**ALWAYS store private keys in:**
- ✅ Hardware wallets (Ledger, Trezor)
- ✅ Encrypted key management systems
- ✅ Secure vault services (AWS KMS, HashiCorp Vault)

---

## 📊 Expected Output

When you run `deploy_unrestricted_network.py`, you'll see:

```
████████████████████████████████████████████████████████████████████████████████
█                                                                              █
█                NEXUS AGI - UNRESTRICTED NETWORK DEPLOYMENT                   █
█                                                                              █
████████████████████████████████████████████████████████████████████████████████

================================================================================
STEP 1: CHECKING BITCOIN WALLET BALANCES
================================================================================

🌐 Connecting to Bitcoin MAINNET...

💰 Checking PRIMARY wallet:
   Address: bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass
   ✅ Balance: 0.00123456 BTC
   💎 Satoshis: 123,456
   📊 Transactions: 5

💰 Checking SECONDARY wallet:
   Address: bc1q8z6z78dy5squapjpkeruem98jcezsw37hnae6q...
   ✅ Balance: 0.00000000 BTC
   💎 Satoshis: 0
   📊 Transactions: 0

💰 Checking LIGHTNING wallet:
   Address: bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
   ✅ Balance: 0.00000000 BTC
   💎 Satoshis: 0
   📊 Transactions: 0

================================================================================
💰 TOTAL MAINNET BALANCE: 0.00123456 BTC
================================================================================

================================================================================
STEP 2: TESTING NETWORK CONNECTIVITY
================================================================================

🔍 Testing MAINNET...
   ✅ Connected successfully
   📊 Block height: 876,543
   🔗 Latest block: 00000000000000000001234567890abc...
   📈 Mempool: 12,345 transactions

🔍 Testing TESTNET...
   ✅ Connected successfully
   📊 Block height: 2,543,210
   🔗 Latest block: 00000000000000012345678abc...
   📈 Mempool: 234 transactions

🔍 Testing SIGNET...
   ✅ Connected successfully
   📊 Block height: 189,456
   🔗 Latest block: 00000000000000abcd123...

================================================================================
STEP 3: TESTING INFINITE DOMAIN NETWORK
================================================================================

🧠 Generating domain knowledge base...
Generated 14821 domains through exponential expansion
   ✅ Generated 100 domains
   📚 Sample domains: ['quantum_physics', 'astrophysics', ...]

🌐 Creating network node...
Created unrestricted node DeploymentTest_Node with 10 domains
   ✅ Node created with 10 domains
   🧠 Consciousness: {'self_awareness': 0.65, ...}

🔬 Testing problem solving...
   ✅ Solution quality: 0.827
   💡 Insights generated: 3

================================================================================
DEPLOYMENT REPORT
================================================================================

📊 DEPLOYMENT SUMMARY:
--------------------------------------------------------------------------------

💰 WALLET BALANCES:
   Total BTC: 0.00123456 BTC
   Approximate USD: $117.28

🌐 NETWORK CONNECTIVITY:
   ✅ MAINNET: connected
      Block height: 876,543
   ✅ TESTNET: connected
      Block height: 2,543,210
   ✅ SIGNET: connected
      Block height: 189,456

🧠 INFINITE DOMAIN NETWORK:
   ✅ Operational
   📚 Domains: 100
   🎯 Quality: 0.827

================================================================================
✅ DEPLOYMENT SUCCESSFUL - ALL SYSTEMS OPERATIONAL
================================================================================

📝 Report saved to: deployment_report_20260113_143022.json

████████████████████████████████████████████████████████████████████████████████
█                                                                              █
█                       DEPLOYMENT TEST COMPLETE                               █
█                                                                              █
████████████████████████████████████████████████████████████████████████████████
```

---

## 🔄 Continuous Monitoring

### Systemd Service (Linux)

Create `/etc/systemd/system/nexus-wallet-monitor.service`:

```ini
[Unit]
Description=NEXUS AGI Wallet Monitor
After=network.target

[Service]
Type=simple
User=nexus
WorkingDirectory=/opt/nexus_agi
ExecStart=/usr/bin/python3 -c "from bitcoin_network_connector import BitcoinNetworkConnector, BitcoinNetwork; import time; connector = BitcoinNetworkConnector(BitcoinNetwork.MAINNET); [connector.check_configured_wallets() or time.sleep(300) for _ in iter(int, 1)]"
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable nexus-wallet-monitor
sudo systemctl start nexus-wallet-monitor
sudo systemctl status nexus-wallet-monitor
```

---

### Cron Job (All Unix Systems)

```bash
# Edit crontab
crontab -e

# Add this line (check every 5 minutes)
*/5 * * * * cd /path/to/nexus_agi && python3 -c "from bitcoin_network_connector import BitcoinNetworkConnector, BitcoinNetwork; connector = BitcoinNetworkConnector(BitcoinNetwork.MAINNET); connector.check_configured_wallets()" >> /var/log/nexus-wallet.log 2>&1
```

---

## 🐛 Troubleshooting

### Connection Refused

**Problem:** `Connection refused` or `Cannot connect`

**Solution:**
```bash
# Check internet connectivity
ping -c 3 blockstream.info

# Check DNS resolution
nslookup blockstream.info

# Try alternative API
python3 -c "from bitcoin_network_connector import BitcoinNetworkConnector, BitcoinNetwork; connector = BitcoinNetworkConnector(BitcoinNetwork.MAINNET); print(connector.get_block_height('mempool'))"
```

---

### Proxy Restrictions (403 Forbidden)

**Problem:** `403 Forbidden` or `Proxy Error`

**Solution:**
```bash
# Clear proxy settings
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# Or configure proxy if needed
export http_proxy="http://proxy.example.com:8080"
export https_proxy="http://proxy.example.com:8080"
```

---

### Module Not Found

**Problem:** `ModuleNotFoundError: No module named 'requests'`

**Solution:**
```bash
# Install all dependencies
pip3 install --upgrade pip
pip3 install requests numpy networkx bitcoinlib web3

# Verify installation
python3 -c "import requests, numpy, networkx, bitcoinlib, web3; print('All modules installed!')"
```

---

## 📞 Quick Commands Reference

```bash
# Check all wallet balances
python3 -c "from bitcoin_network_connector import BitcoinNetworkConnector, BitcoinNetwork; BitcoinNetworkConnector(BitcoinNetwork.MAINNET).check_configured_wallets()"

# Get current block height
python3 -c "from bitcoin_network_connector import BitcoinNetworkConnector, BitcoinNetwork; print(BitcoinNetworkConnector(BitcoinNetwork.MAINNET).get_block_height())"

# Check specific address
python3 -c "from bitcoin_network_connector import BitcoinNetworkConnector, BitcoinNetwork; connector = BitcoinNetworkConnector(BitcoinNetwork.MAINNET); print(f'{connector.get_address_balance(\"bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass\") / 100_000_000:.8f} BTC')"

# Get mempool status
python3 -c "from bitcoin_network_connector import BitcoinNetworkConnector, BitcoinNetwork; import json; print(json.dumps(BitcoinNetworkConnector(BitcoinNetwork.MAINNET).get_mempool_status(), indent=2))"

# Full deployment test
python3 deploy_unrestricted_network.py
```

---

## 🎯 Summary

**To check your wallet balances on unrestricted network:**

1. **Fastest way:**
   ```bash
   cd /path/to/nexus_agi
   pip3 install requests numpy networkx bitcoinlib web3
   python3 deploy_unrestricted_network.py
   ```

2. **Docker way:**
   ```bash
   docker build -f Dockerfile.bitcoin -t nexus-bitcoin .
   docker run --rm nexus-bitcoin
   ```

3. **Production way:**
   ```bash
   docker-compose -f docker-compose.bitcoin.yml up -d
   docker-compose logs -f nexus-wallet-monitor
   ```

**Your balances will be displayed immediately!** 💰

---

*Last updated: 2026-01-13*
*NEXUS AGI - Unrestricted Network Deployment*
