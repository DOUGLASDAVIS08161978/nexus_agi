#!/bin/bash
#
# Nexus AGI - Bitcoin Payment Setup
# Quick setup for receiving payments in your Bitcoin wallet
#

set -e

echo ""
echo "================================================================================"
echo "💰 NEXUS AGI - BITCOIN PAYMENT SETUP 💰"
echo "================================================================================"
echo ""
echo "This script will help you configure Bitcoin payments for your Nexus AGI API"
echo ""

# Get user's Bitcoin wallet address
read -p "Enter your Bitcoin wallet address (ending in WASS): " BTC_WALLET
echo ""

# Validate it looks like a Bitcoin address
if [[ ! $BTC_WALLET =~ ^(bc1|1|3) ]]; then
    echo "❌ That doesn't look like a valid Bitcoin address"
    echo "Bitcoin addresses start with: bc1 (SegWit), 1 (P2PKH), or 3 (P2SH)"
    exit 1
fi

echo "✅ Bitcoin wallet: $BTC_WALLET"
echo ""

# Choose payment method
echo "Choose your Bitcoin payment processor:"
echo ""
echo "  [1] BTCPay Server (RECOMMENDED - Free, self-hosted, 0% fees)"
echo "  [2] Coinbase Commerce (Easy setup, ~1% fees)"
echo "  [3] OpenNode (Lightning Network, instant payments)"
echo "  [4] Manual setup (I'll configure it myself)"
echo ""
read -p "Enter choice [1-4]: " PAYMENT_CHOICE

case $PAYMENT_CHOICE in
    1)
        echo ""
        echo "📡 BTCPay Server Setup"
        echo "─────────────────────"
        read -p "Enter your BTCPay Server URL (e.g., https://btcpay.yourdomain.com): " BTCPAY_URL
        read -p "Enter your BTCPay API Key: " BTCPAY_KEY
        read -p "Enter your BTCPay Store ID: " BTCPAY_STORE

        PAYMENT_CONFIG="# BTCPay Server Configuration
BITCOIN_WALLET_ADDRESS=$BTC_WALLET
BTCPAY_SERVER_URL=$BTCPAY_URL
BTCPAY_API_KEY=$BTCPAY_KEY
BTCPAY_STORE_ID=$BTCPAY_STORE
PAYMENT_PROCESSOR=btcpay"
        ;;
    2)
        echo ""
        echo "🏦 Coinbase Commerce Setup"
        echo "──────────────────────────"
        echo "1. Sign up at: https://commerce.coinbase.com"
        echo "2. Get your API key from Settings"
        echo "3. Set withdrawal address to: $BTC_WALLET"
        echo ""
        read -p "Enter your Coinbase Commerce API Key: " COINBASE_KEY

        PAYMENT_CONFIG="# Coinbase Commerce Configuration
BITCOIN_WALLET_ADDRESS=$BTC_WALLET
COINBASE_COMMERCE_API_KEY=$COINBASE_KEY
PAYMENT_PROCESSOR=coinbase"
        ;;
    3)
        echo ""
        echo "⚡ OpenNode Setup"
        echo "─────────────────"
        read -p "Enter your OpenNode API Key: " OPENNODE_KEY

        PAYMENT_CONFIG="# OpenNode Configuration
BITCOIN_WALLET_ADDRESS=$BTC_WALLET
OPENNODE_API_KEY=$OPENNODE_KEY
PAYMENT_PROCESSOR=opennode
ACCEPT_LIGHTNING=true"
        ;;
    4)
        echo ""
        echo "⚙️  Manual Configuration"
        echo "───────────────────────"

        PAYMENT_CONFIG="# Manual Bitcoin Configuration
BITCOIN_WALLET_ADDRESS=$BTC_WALLET
PAYMENT_PROCESSOR=manual"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# Pricing configuration
echo ""
echo "💵 Set Your Pricing"
echo "───────────────────"
echo ""
echo "Current Bitcoin price: ~\$50,000 (check current price at bitcoinprice.com)"
echo ""

# Calculate BTC amounts
read -p "Price per API call in USD [$0.05]: " PRICE_API
PRICE_API=${PRICE_API:-0.05}

read -p "Price per consultation in USD [$100]: " PRICE_CONSULT
PRICE_CONSULT=${PRICE_CONSULT:-100}

read -p "Price per analysis in USD [$100]: " PRICE_ANALYSIS
PRICE_ANALYSIS=${PRICE_ANALYSIS:-100}

# Simple conversion (assuming $50k BTC price)
BTC_API=$(echo "scale=8; $PRICE_API / 50000" | bc)
BTC_CONSULT=$(echo "scale=8; $PRICE_CONSULT / 50000" | bc)
BTC_ANALYSIS=$(echo "scale=8; $PRICE_ANALYSIS / 50000" | bc)

PRICING_CONFIG="
# Pricing Configuration (in USD and BTC)
PRICE_PER_API_CALL_USD=$PRICE_API
PRICE_PER_API_CALL_BTC=$BTC_API

PRICE_CONSULTATION_USD=$PRICE_CONSULT
PRICE_CONSULTATION_BTC=$BTC_CONSULT

PRICE_ANALYSIS_USD=$PRICE_ANALYSIS
PRICE_ANALYSIS_BTC=$BTC_ANALYSIS

# Auto-conversion
BTC_PRICE_USD=50000  # Update this regularly or use API
"

# Auto-withdrawal settings
echo ""
echo "🔄 Auto-Withdrawal Settings"
echo "───────────────────────────"
echo ""
read -p "Automatically withdraw to your wallet? [Y/n]: " AUTO_WITHDRAW
AUTO_WITHDRAW=${AUTO_WITHDRAW:-Y}

if [[ $AUTO_WITHDRAW =~ ^[Yy] ]]; then
    read -p "Minimum balance before withdrawal (BTC) [0.01]: " MIN_BALANCE
    MIN_BALANCE=${MIN_BALANCE:-0.01}

    echo "Withdrawal schedule:"
    echo "  [1] Instant (after each payment)"
    echo "  [2] Daily"
    echo "  [3] Weekly"
    read -p "Choose [1-3]: " WITHDRAW_SCHEDULE

    case $WITHDRAW_SCHEDULE in
        1) SCHEDULE="instant" ;;
        2) SCHEDULE="daily" ;;
        3) SCHEDULE="weekly" ;;
        *) SCHEDULE="daily" ;;
    esac

    WITHDRAW_CONFIG="
# Auto-Withdrawal Configuration
AUTO_WITHDRAW_TO_WALLET=true
MIN_BALANCE_BEFORE_WITHDRAW=$MIN_BALANCE
WITHDRAW_SCHEDULE=$SCHEDULE
WITHDRAW_TO_ADDRESS=$BTC_WALLET
"
else
    WITHDRAW_CONFIG="
# Auto-Withdrawal Configuration
AUTO_WITHDRAW_TO_WALLET=false
"
fi

# Security settings
ADMIN_KEY=$(openssl rand -hex 32)

SECURITY_CONFIG="
# Security Configuration
ADMIN_KEY=$ADMIN_KEY
API_RATE_LIMIT=100
REQUIRE_PAYMENT_CONFIRMATION=3  # Bitcoin confirmations
SESSION_SECRET=$(openssl rand -hex 32)
"

# Create .env file
echo ""
echo "📝 Creating .env file..."

cat > .env << EOF
# ===============================================================================
# NEXUS AGI - BITCOIN PAYMENT CONFIGURATION
# Generated: $(date)
# ===============================================================================

$PAYMENT_CONFIG

$PRICING_CONFIG

$WITHDRAW_CONFIG

$SECURITY_CONFIG

# API Configuration
API_URL=http://localhost:8000
ENVIRONMENT=production

# Logging
LOG_LEVEL=INFO
LOG_FILE=nexus_payments.log

# ===============================================================================
EOF

chmod 600 .env

echo "✅ Configuration saved to .env"
echo ""

# Create a summary
echo "================================================================================"
echo "📊 CONFIGURATION SUMMARY"
echo "================================================================================"
echo ""
echo "💰 Bitcoin Wallet: $BTC_WALLET"
echo ""
echo "💵 Pricing:"
echo "   - API Call: \$$PRICE_API (≈ $BTC_API BTC)"
echo "   - Consultation: \$$PRICE_CONSULT (≈ $BTC_CONSULT BTC)"
echo "   - Analysis: \$$PRICE_ANALYSIS (≈ $BTC_ANALYSIS BTC)"
echo ""
echo "🔐 Admin Key: $ADMIN_KEY"
echo "   (Save this! You'll need it for admin access)"
echo ""

if [[ $AUTO_WITHDRAW =~ ^[Yy] ]]; then
    echo "🔄 Auto-Withdrawal: Enabled ($SCHEDULE)"
    echo "   Minimum balance: $MIN_BALANCE BTC"
else
    echo "🔄 Auto-Withdrawal: Disabled (manual withdrawals)"
fi

echo ""
echo "================================================================================"
echo "🚀 NEXT STEPS"
echo "================================================================================"
echo ""
echo "1. Start the API server:"
echo "   python3 realworld_api_deployment.py"
echo ""
echo "2. Test payment creation:"
echo "   curl -X POST http://localhost:8000/api/v1/create-payment \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"service\":\"test\",\"amount_usd\":1}'"
echo ""
echo "3. Access admin dashboard:"
echo "   curl http://localhost:8000/admin/stats \\"
echo "     -H 'admin-key: $ADMIN_KEY'"
echo ""
echo "4. Deploy to production (see REAL_INCOME_DEPLOYMENT.md)"
echo ""
echo "================================================================================"
echo "✅ BITCOIN PAYMENT SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "⚡ Your Nexus AGI is ready to accept Bitcoin payments!"
echo "💰 All income will flow to: $BTC_WALLET"
echo ""
echo "Read REAL_INCOME_DEPLOYMENT.md for full deployment guide."
echo ""
