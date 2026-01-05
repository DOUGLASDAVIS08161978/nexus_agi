#!/bin/bash
# Bitcoin to Ethereum Bridge Transfer Script
# Example script to execute bridge transfers

set -e

echo "=========================================="
echo "Bitcoin to Ethereum Bridge Transfer"
echo "=========================================="
echo ""

# Load environment variables
if [ -f .env.bridge ]; then
    export $(cat .env.bridge | grep -v '^#' | xargs)
    echo "✓ Configuration loaded from .env.bridge"
else
    echo "❌ Error: .env.bridge file not found"
    echo "Copy .env.bridge.example to .env.bridge and configure it"
    exit 1
fi

# Validate required variables
if [ -z "$ETH_PRIVATE_KEY" ]; then
    echo "❌ Error: ETH_PRIVATE_KEY not set in .env.bridge"
    exit 1
fi

if [ -z "$ETH_DESTINATION_ADDRESS" ]; then
    echo "❌ Error: ETH_DESTINATION_ADDRESS not set in .env.bridge"
    exit 1
fi

echo "Configuration:"
echo "  Ethereum Network: ${ETH_NETWORK:-mainnet}"
echo "  Bitcoin Network: ${BTC_NETWORK:-mainnet}"
echo "  Destination: $ETH_DESTINATION_ADDRESS"
echo ""

# Install dependencies if needed
if ! python3 -c "import web3" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r bridge_requirements.txt
fi

# Check balance first
echo "Checking Ethereum balance..."
python3 bridge_orchestrator.py \
    --eth-address "$ETH_DESTINATION_ADDRESS" \
    --network "${ETH_NETWORK:-mainnet}" \
    --check-balance

echo ""
read -p "Continue with transfer? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Transfer cancelled"
    exit 0
fi

# Get amount to transfer
read -p "Enter amount in BTC to transfer: " amount

if [ -z "$amount" ]; then
    echo "❌ Error: Amount is required"
    exit 1
fi

# Execute transfer
echo ""
echo "Executing bridge transfer..."
python3 bridge_orchestrator.py \
    --btc-address "${BTC_SOURCE_ADDRESS}" \
    --eth-address "$ETH_DESTINATION_ADDRESS" \
    --amount "$amount" \
    --network "${ETH_NETWORK:-mainnet}" \
    --email "${NOTIFICATION_EMAIL}" \
    --btc-tx-hash "${BTC_TX_HASH}"

echo ""
echo "✓ Transfer script completed"
