#!/bin/bash
# WTBTC Mainnet Deployment - One Command
# Pulls latest code, compiles, deploys, verifies, and publishes

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🪙 WTBTC DEPLOYMENT - WRAPPED BITCOIN 1:1 PEG"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Step 1: Pull latest code
echo ""
echo "📥 STEP 1: Pulling latest code from GitHub..."
git pull origin claude/add-web3-onboard-qFwXx || {
    echo "⚠️  Git pull failed, continuing anyway..."
}

# Step 2: Install dependencies (if needed)
echo ""
echo "📦 STEP 2: Checking dependencies..."
if [ ! -d "node_modules/solc" ] || [ ! -d "node_modules/ethers" ]; then
    echo "Installing dependencies..."
    npm install --silent
else
    echo "✅ Dependencies already installed"
fi

# Step 3: Deploy WTBTC
echo ""
echo "🚀 STEP 3: Deploying WTBTC to Mainnet..."
echo ""

node deploy_wtbtc.js mainnet

# Done
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ WTBTC DEPLOYMENT COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
