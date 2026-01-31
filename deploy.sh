#!/bin/bash

# Autonomous WalletManager Deployment Script
# This script handles all deployment steps automatically

echo "======================================================================="
echo "🚀 AUTONOMOUS WALLETMANAGER DEPLOYMENT"
echo "======================================================================="

# Set environment to bypass SSL issues in this environment
export NODE_TLS_REJECT_UNAUTHORIZED=0

# Determine network (default to mainnet)
NETWORK="${1:-mainnet}"

echo ""
echo "📡 Network: $NETWORK"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ ERROR: .env file not found!"
    echo "Please ensure .env exists with PRIVATE_KEY configured"
    exit 1
fi

# Run deployment using hardhat from node_modules
echo "⏳ Starting deployment..."
echo ""

# Use node to run hardhat directly
node -e "require('hardhat/internal/cli/cli').main(['run', 'scripts/deploy_and_verify.js', '--network', '$NETWORK'])"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "✅ DEPLOYMENT SUCCESSFUL!"
    echo "======================================================================="
else
    echo ""
    echo "======================================================================="
    echo "❌ DEPLOYMENT FAILED!"
    echo "======================================================================="
    exit $EXIT_CODE
fi
